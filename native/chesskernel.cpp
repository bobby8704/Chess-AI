// chesskernel: native kernel for the chess engine's leaf evaluation.
//
// Stage 1 scope (native/PORT.md): bitboard position + LEGAL move generation.
// The quiescence search needs captures+promotions to recurse on and a
// has-any-legal-move existence check at every node (mate/stalemate), so full
// legal movegen — castling included — is required even though only captures
// are searched. Conventions match python-chess: square 0 = a1, bit i = square
// i, white pawns push +8. Every function takes the position as plain integers
// unpacked from python-chess internals — no FEN strings on the hot path.
//
// Build:  .venv/Scripts/python.exe native/setup_native.py build_ext --inplace
// Test:   .venv/Scripts/python.exe native/test_native.py

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace py = pybind11;
using u64 = uint64_t;

enum Piece { P = 0, N = 1, B = 2, R = 3, Q = 4, K = 5 };

struct Pos {
    u64 pc[2][6];   // [color][piece], color 0 = white
    u64 occ[2];
    u64 all;
    int stm;        // side to move: 0 white, 1 black
    int ep;         // en-passant target square, -1 if none
    u64 castling;   // bitboard of rook home squares with rights (python-chess)
};

struct Move {
    int from, to;
    int promo;      // 0 none, else Piece (N/B/R/Q as python-chess ints 2..5)
};

// python-chess piece ints: PAWN=1 KNIGHT=2 BISHOP=3 ROOK=4 QUEEN=5 KING=6.
// We store 0..5 internally; promo is exported in python-chess numbering.
static inline int to_pychess_piece(int p) { return p + 1; }

// ---------------------------------------------------------------------------
// Attack tables
// ---------------------------------------------------------------------------

static u64 KNIGHT_ATK[64];
static u64 KING_ATK[64];
static u64 PAWN_ATK[2][64];    // [color][from] — squares a pawn of color attacks

static inline int file_of(int sq) { return sq & 7; }
static inline int rank_of(int sq) { return sq >> 3; }
static inline u64 bit(int sq) { return 1ULL << sq; }

static void add_delta(u64 &mask, int sq, int df, int dr) {
    int f = file_of(sq) + df, r = rank_of(sq) + dr;
    if (f >= 0 && f < 8 && r >= 0 && r < 8) mask |= bit(r * 8 + f);
}

static bool init_tables() {
    for (int sq = 0; sq < 64; ++sq) {
        u64 n = 0, k = 0;
        const int nd[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
        for (auto &d : nd) add_delta(n, sq, d[0], d[1]);
        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr)
                if (df || dr) add_delta(k, sq, df, dr);
        KNIGHT_ATK[sq] = n;
        KING_ATK[sq] = k;
        u64 pw = 0, pb = 0;
        add_delta(pw, sq, -1, 1);  add_delta(pw, sq, 1, 1);
        add_delta(pb, sq, -1, -1); add_delta(pb, sq, 1, -1);
        PAWN_ATK[0][sq] = pw;
        PAWN_ATK[1][sq] = pb;
    }
    return true;
}
static const bool TABLES_READY = init_tables();

// Sliding attacks by directional stepping. Loop-based, not magics: clarity and
// correctness first; this is already orders of magnitude beyond python-chess.
static u64 slide(int sq, u64 occ, const int (*dirs)[2], int ndirs) {
    u64 atk = 0;
    for (int i = 0; i < ndirs; ++i) {
        int f = file_of(sq), r = rank_of(sq);
        for (;;) {
            f += dirs[i][0]; r += dirs[i][1];
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            int t = r * 8 + f;
            atk |= bit(t);
            if (occ & bit(t)) break;
        }
    }
    return atk;
}

static const int DIAG[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
static const int ORTH[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

static inline u64 bishop_atk(int sq, u64 occ) { return slide(sq, occ, DIAG, 4); }
static inline u64 rook_atk(int sq, u64 occ)   { return slide(sq, occ, ORTH, 4); }

// Is `sq` attacked by side `by`?
static bool square_attacked(const Pos &p, int sq, int by) {
    if (PAWN_ATK[by ^ 1][sq] & p.pc[by][P]) return true;   // reverse pawn lookup
    if (KNIGHT_ATK[sq] & p.pc[by][N]) return true;
    if (KING_ATK[sq] & p.pc[by][K]) return true;
    u64 diag = bishop_atk(sq, p.all);
    if (diag & (p.pc[by][B] | p.pc[by][Q])) return true;
    u64 orth = rook_atk(sq, p.all);
    if (orth & (p.pc[by][R] | p.pc[by][Q])) return true;
    return false;
}

static inline int king_sq(const Pos &p, int c) {
    u64 k = p.pc[c][K];
    // k is never 0 in a legal position; ctz is safe.
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanForward64(&idx, k);
    return (int)idx;
#else
    return __builtin_ctzll(k);
#endif
}

static inline bool in_check_(const Pos &p, int c) {
    // Kingless boards are reachable in ILLEGAL positions (python-chess lets
    // you capture a king that was left attacked with the wrong side to move,
    // and its is_check() then reports False). Real play can never get here,
    // but the differential harness proved BitScanForward on an empty king
    // bitboard is garbage-in-garbage-out, so match python-chess exactly.
    if (!p.pc[c][K]) return false;
    return square_attacked(p, king_sq(p, c), c ^ 1);
}

// ---------------------------------------------------------------------------
// Make move (copy-make)
// ---------------------------------------------------------------------------

static int piece_on(const Pos &p, int c, int sq) {
    u64 b = bit(sq);
    for (int pt = 0; pt < 6; ++pt)
        if (p.pc[c][pt] & b) return pt;
    return -1;
}

static Pos make_move(const Pos &p, const Move &m) {
    Pos n = p;
    int us = p.stm, them = us ^ 1;
    u64 fromb = bit(m.from), tob = bit(m.to);
    int pt = piece_on(p, us, m.from);

    // Capture (including the EP special case where the victim is not on `to`).
    if (n.occ[them] & tob) {
        int vic = piece_on(p, them, m.to);
        n.pc[them][vic] &= ~tob;
        n.occ[them] &= ~tob;
        n.castling &= ~tob;                     // captured a rook with rights
    } else if (pt == P && m.to == p.ep && p.ep >= 0) {
        int vic_sq = (us == 0) ? m.to - 8 : m.to + 8;
        n.pc[them][P] &= ~bit(vic_sq);
        n.occ[them] &= ~bit(vic_sq);
    }

    // Move the piece (with promotion replacement).
    n.pc[us][pt] &= ~fromb;
    if (m.promo) n.pc[us][m.promo - 1] |= tob;  // promo arrives python-chess-numbered
    else         n.pc[us][pt] |= tob;
    n.occ[us] = (n.occ[us] & ~fromb) | tob;

    // Castling: the king moved two files — move the rook too.
    if (pt == K) {
        n.castling &= (us == 0) ? ~0xFFULL : ~0xFF00000000000000ULL;
        int df = file_of(m.to) - file_of(m.from);
        if (df == 2) {          // king side: rook h->f
            int rf = m.to + 1, rt = m.to - 1;
            n.pc[us][R] &= ~bit(rf); n.pc[us][R] |= bit(rt);
            n.occ[us] &= ~bit(rf);   n.occ[us] |= bit(rt);
        } else if (df == -2) {  // queen side: rook a->d
            int rf = m.to - 2, rt = m.to + 1;
            n.pc[us][R] &= ~bit(rf); n.pc[us][R] |= bit(rt);
            n.occ[us] &= ~bit(rf);   n.occ[us] |= bit(rt);
        }
    }
    n.castling &= ~fromb;                       // rook left home, or king square

    // En-passant target: set only on a double pawn push.
    n.ep = -1;
    if (pt == P) {
        int d = m.to - m.from;
        if (d == 16)  n.ep = m.from + 8;
        if (d == -16) n.ep = m.from - 8;
    }

    n.all = n.occ[0] | n.occ[1];
    n.stm = them;
    return n;
}

// ---------------------------------------------------------------------------
// Move generation (pseudo-legal, filtered to legal by make + in-check)
// ---------------------------------------------------------------------------

static void push_pawn_targets(std::vector<Move> &out, int from, int to, int us) {
    int last = (us == 0) ? 7 : 0;
    if (rank_of(to) == last) {
        // python-chess numbering: QUEEN=5, ROOK=4, BISHOP=3, KNIGHT=2
        out.push_back({from, to, 5});
        out.push_back({from, to, 4});
        out.push_back({from, to, 3});
        out.push_back({from, to, 2});
    } else {
        out.push_back({from, to, 0});
    }
}

static void gen_pseudo(const Pos &p, std::vector<Move> &out) {
    int us = p.stm, them = us ^ 1;
    u64 own = p.occ[us], opp = p.occ[them], all = p.all;

    // Pawns
    u64 pawns = p.pc[us][P];
    int fwd = (us == 0) ? 8 : -8;
    int start_rank = (us == 0) ? 1 : 6;
    for (u64 bb = pawns; bb; bb &= bb - 1) {
#ifdef _MSC_VER
        unsigned long idx; _BitScanForward64(&idx, bb); int from = (int)idx;
#else
        int from = __builtin_ctzll(bb);
#endif
        int one = from + fwd;
        if (!(all & bit(one))) {
            push_pawn_targets(out, from, one, us);
            if (rank_of(from) == start_rank && !(all & bit(one + fwd)))
                out.push_back({from, one + fwd, 0});
        }
        u64 atk = PAWN_ATK[us][from];
        for (u64 t = atk & opp; t; t &= t - 1) {
#ifdef _MSC_VER
            unsigned long ti; _BitScanForward64(&ti, t); int to = (int)ti;
#else
            int to = __builtin_ctzll(t);
#endif
            push_pawn_targets(out, from, to, us);
        }
        if (p.ep >= 0 && (atk & bit(p.ep)))
            out.push_back({from, p.ep, 0});
    }

    // Knights, bishops, rooks, queens, king
    auto gen_from_mask = [&](u64 pieces, auto attacks) {
        for (u64 bb = pieces; bb; bb &= bb - 1) {
#ifdef _MSC_VER
            unsigned long idx; _BitScanForward64(&idx, bb); int from = (int)idx;
#else
            int from = __builtin_ctzll(bb);
#endif
            u64 targets = attacks(from) & ~own;
            for (u64 t = targets; t; t &= t - 1) {
#ifdef _MSC_VER
                unsigned long ti; _BitScanForward64(&ti, t); int to = (int)ti;
#else
                int to = __builtin_ctzll(t);
#endif
                out.push_back({from, to, 0});
            }
        }
    };
    gen_from_mask(p.pc[us][N], [&](int s) { return KNIGHT_ATK[s]; });
    gen_from_mask(p.pc[us][B], [&](int s) { return bishop_atk(s, all); });
    gen_from_mask(p.pc[us][R], [&](int s) { return rook_atk(s, all); });
    gen_from_mask(p.pc[us][Q], [&](int s) { return bishop_atk(s, all) | rook_atk(s, all); });
    gen_from_mask(p.pc[us][K], [&](int s) { return KING_ATK[s]; });

    // Castling (standard chess). Rights bitboard holds rook home squares.
    if (!p.pc[us][K]) return;                   // kingless: nothing to castle
    int ks = king_sq(p, us);
    int home = (us == 0) ? 4 : 60;              // e1 / e8
    if (ks == home && !in_check_(p, us)) {
        int rank_base = (us == 0) ? 0 : 56;
        // King side: rook on h-file, f and g empty, e/f/g not attacked.
        if (p.castling & bit(rank_base + 7)) {
            if (!(all & (bit(rank_base + 5) | bit(rank_base + 6)))
                && !square_attacked(p, rank_base + 5, them)
                && !square_attacked(p, rank_base + 6, them))
                out.push_back({home, rank_base + 6, 0});
        }
        // Queen side: rook on a-file, b/c/d empty, c/d/e not attacked.
        if (p.castling & bit(rank_base + 0)) {
            if (!(all & (bit(rank_base + 1) | bit(rank_base + 2) | bit(rank_base + 3)))
                && !square_attacked(p, rank_base + 2, them)
                && !square_attacked(p, rank_base + 3, them))
                out.push_back({home, rank_base + 2, 0});
        }
    }
}

static std::vector<Move> gen_legal(const Pos &p) {
    std::vector<Move> pseudo, legal;
    pseudo.reserve(64);
    legal.reserve(48);
    gen_pseudo(p, pseudo);
    for (const Move &m : pseudo) {
        Pos n = make_move(p, m);
        if (!in_check_(n, p.stm)) legal.push_back(m);
    }
    return legal;
}

static bool has_legal_move(const Pos &p) {
    std::vector<Move> pseudo;
    pseudo.reserve(64);
    gen_pseudo(p, pseudo);
    for (const Move &m : pseudo) {
        Pos n = make_move(p, m);
        if (!in_check_(n, p.stm)) return true;
    }
    return false;
}

// A move is a "quiescence move" iff python-chess would say
// board.is_capture(m) or m.promotion — capture includes en passant.
static inline bool is_qmove(const Pos &p, const Move &m) {
    if (m.promo) return true;
    if (p.occ[p.stm ^ 1] & bit(m.to)) return true;
    int pt = piece_on(p, p.stm, m.from);
    return pt == P && p.ep >= 0 && m.to == p.ep;
}

// ---------------------------------------------------------------------------
// Stage 1b: _evaluate_raw (evaluation.py:736) — the quiescence stand-pat.
// 28.4% of every move, ~80k calls (bench/results/profile_move_1300.json).
// Every table is verbatim from evaluation.py; every arithmetic quirk — the
// (int) truncation toward zero, the king PST phase blend order, the per-king
// truncation inside the material loop — replicates Python exactly, because
// the differential gate demands integer equality on every position. The
// build uses /fp:strict (-ffp-contract=off) so the compiler cannot fuse the
// blend into FMA and change the double result by an ulp.
// ---------------------------------------------------------------------------

static inline int get_lsb(u64 b) {
#ifdef _MSC_VER
    unsigned long i; _BitScanForward64(&i, b); return (int)i;
#else
    return __builtin_ctzll(b);
#endif
}

static inline int popcnt(u64 b) {
#ifdef _MSC_VER
    return (int)__popcnt64(b);
#else
    return __builtin_popcountll(b);
#endif
}

static const int PIECE_CP[6] = {100, 320, 330, 500, 900, 0};

// fmt matches evaluation.py:45-120 line for line.
static const int PST_PAWN_T[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
};
static const int PST_KNIGHT_T[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
};
static const int PST_BISHOP_T[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
};
static const int PST_ROOK_T[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
};
static const int PST_QUEEN_T[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
};
static const int PST_KING_MG_T[64] = {
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
};
static const int PST_KING_EG_T[64] = {
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
};
static const int CENTER_DIST[64] = {
    3, 2, 2, 2, 2, 2, 2, 3,
    2, 1, 1, 1, 1, 1, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 1, 1, 1, 1, 1, 2,
    3, 2, 2, 2, 2, 2, 2, 3,
};

static const u64 FILE_A = 0x0101010101010101ULL;
static const u64 BB_LIGHT = 0x55aa55aa55aa55aaULL;
static const u64 BB_DARK  = 0xaa55aa55aa55aa55ULL;

// python-chess has_insufficient_material, read from the installed source
// (chess/__init__.py:2117) rather than remembered: the knight clause requires
// the OPPONENT to hold nothing beyond kings and queens (rooks break it), and
// the bishop clause looks at ALL bishops on the board sharing a square colour
// plus a global no-pawns no-knights condition.
static bool side_insufficient(const Pos &p, int c) {
    if (p.pc[c][P] | p.pc[c][R] | p.pc[c][Q]) return false;
    if (p.pc[c][N]) {
        u64 kings_q = p.pc[0][K] | p.pc[1][K] | p.pc[0][Q] | p.pc[1][Q];
        return popcnt(p.occ[c]) <= 2 && !(p.occ[c ^ 1] & ~kings_q);
    }
    if (p.pc[c][B]) {
        u64 bishops = p.pc[0][B] | p.pc[1][B];
        bool same_color = !(bishops & BB_DARK) || !(bishops & BB_LIGHT);
        u64 pawns = p.pc[0][P] | p.pc[1][P];
        u64 knights = p.pc[0][N] | p.pc[1][N];
        return same_color && !pawns && !knights;
    }
    return true;
}

static double game_phase(const Pos &p) {
    int npm = 0;
    for (int c = 0; c < 2; ++c)
        npm += popcnt(p.pc[c][N]) * 320 + popcnt(p.pc[c][B]) * 330
             + popcnt(p.pc[c][R]) * 500 + popcnt(p.pc[c][Q]) * 900;
    if (npm >= 4000) return 0.0;
    if (npm <= 1000) return 1.0;
    return 1.0 - (npm - 1000) / 3000.0;
}

static int pst_value(int pt, int sq, int c, double egw) {
    int idx = (c == 0) ? sq : (sq ^ 56);        // chess.square_mirror
    if (pt == K) {
        // Python: int(mg * (1 - w) + eg * w) — trunc toward zero, exact order.
        double v = PST_KING_MG_T[idx] * (1.0 - egw) + PST_KING_EG_T[idx] * egw;
        return (int)v;
    }
    switch (pt) {
        case P: return PST_PAWN_T[idx];
        case N: return PST_KNIGHT_T[idx];
        case B: return PST_BISHOP_T[idx];
        case R: return PST_ROOK_T[idx];
        case Q: return PST_QUEEN_T[idx];
    }
    return 0;
}

static int pawn_structure(const Pos &p, int c) {
    int score = 0;
    u64 our = p.pc[c][P], opp = p.pc[c ^ 1][P];
    int files_mask = 0;
    for (u64 bb = our; bb; bb &= bb - 1)
        files_mask |= 1 << file_of(get_lsb(bb));

    for (u64 bb = our; bb; bb &= bb - 1) {
        int sq = get_lsb(bb);
        int f = file_of(sq), r = rank_of(sq);
        if (our & (FILE_A << f) & ~bit(sq)) score -= 15;                 // doubled
        bool neighbor = (f > 0 && (files_mask & (1 << (f - 1))))
                     || (f < 7 && (files_mask & (1 << (f + 1))));
        if (!neighbor) score -= 20;                                      // isolated
        u64 span = 0;
        for (int cf = (f > 0 ? f - 1 : 0); cf <= (f < 7 ? f + 1 : 7); ++cf)
            span |= FILE_A << cf;
        u64 ahead = (c == 0)
            ? (r >= 7 ? 0ULL : ~((1ULL << ((r + 1) * 8)) - 1))           // ranks > r
            : ((1ULL << (r * 8)) - 1);                                   // ranks < r
        if (!(opp & span & ahead)) {                                     // passed
            int adv = (c == 0) ? r - 1 : 6 - r;
            score += 20 + adv * 15;
        }
    }
    return score;
}

static int king_safety(const Pos &p, int c) {
    if (!p.pc[c][K]) return 0;
    int ks = king_sq(p, c);
    int score = 0;
    int kf = file_of(ks), kr = rank_of(ks);
    int sr = (c == 0) ? kr + 1 : kr - 1;
    if (sr >= 0 && sr <= 7) {
        for (int f = (kf > 0 ? kf - 1 : 0); f <= (kf < 7 ? kf + 1 : 7); ++f) {
            int sq = sr * 8 + f;
            score += (p.pc[c][P] & bit(sq)) ? 15 : -15;   // shelter pawn or hole
        }
    }
    if (!((p.pc[0][P] | p.pc[1][P]) & (FILE_A << kf))) score -= 30;      // open file
    return score;
}

static int back_rank_safety(const Pos &p, int c, int fullmove) {
    if (!p.pc[c][K]) return 0;
    int ks = king_sq(p, c);
    if (rank_of(ks) != ((c == 0) ? 0 : 7)) return 0;
    if (fullmove < 8) return 0;
    int kf = file_of(ks);
    int er = (c == 0) ? 1 : 6;
    for (int f = (kf > 0 ? kf - 1 : 0); f <= (kf < 7 ? kf + 1 : 7); ++f) {
        int sq = er * 8 + f;
        if (!(p.occ[c] & bit(sq)) && !square_attacked(p, sq, c ^ 1))
            return 0;                                     // has an escape square
    }
    int rq = popcnt(p.pc[c ^ 1][R]) + popcnt(p.pc[c ^ 1][Q]);
    return rq ? -80 * rq : 0;
}

static int checkmate_forcing(const Pos &p, int strong) {
    int weak = strong ^ 1;
    if (p.occ[weak] & ~p.pc[weak][K]) return 0;           // weak side has material
    if (!p.pc[strong][Q] && !p.pc[strong][R]) return 0;   // need a Q or R to mate
    if (!p.pc[weak][K] || !p.pc[strong][K]) return 0;
    int wk = king_sq(p, weak), sk = king_sq(p, strong);
    int score = CENTER_DIST[wk] * 150;
    int kd = std::max(std::abs(file_of(wk) - file_of(sk)),
                      std::abs(rank_of(wk) - rank_of(sk)));
    score += (7 - kd) * 80;
    int moves = 0;
    for (int sq = 0; sq < 64; ++sq) {
        int cd = std::max(std::abs(file_of(wk) - file_of(sq)),
                          std::abs(rank_of(wk) - rank_of(sq)));
        if (cd != 1) continue;
        if (!square_attacked(p, sq, strong) && !(p.occ[weak] & bit(sq)))
            ++moves;                                      // escape or capture square
    }
    score += (8 - moves) * 50;
    return score;
}

// The scoring body, AFTER the terminal checks. qsearch calls this directly:
// by the time stand-pat is computed the search has already established that
// legal moves exist and material is sufficient, exactly as Python's
// _quiescence has before it calls _evaluate_raw — whose own re-checks then
// fall through. Skipping the re-check is identical by construction.
static int evaluate_raw_core(const Pos &p, int fullmove) {
    int score = 0;
    double egw = game_phase(p);
    for (int c = 0; c < 2; ++c) {
        int sign = (c == 0) ? 1 : -1;
        for (int pt = 0; pt < 6; ++pt)
            for (u64 bb = p.pc[c][pt]; bb; bb &= bb - 1) {
                int sq = get_lsb(bb);
                score += sign * (PIECE_CP[pt] + pst_value(pt, sq, c, egw));
            }
    }
    score += pawn_structure(p, 0);
    score -= pawn_structure(p, 1);
    double mgw = 1.0 - egw;
    score += (int)(king_safety(p, 0) * mgw);              // Python int(): trunc to 0
    score -= (int)(king_safety(p, 1) * mgw);
    score += back_rank_safety(p, 0, fullmove);
    score -= back_rank_safety(p, 1, fullmove);
    score += checkmate_forcing(p, 0);
    score -= checkmate_forcing(p, 1);
    if (popcnt(p.pc[0][B]) >= 2) score += 30;
    if (popcnt(p.pc[1][B]) >= 2) score -= 30;
    return (p.stm == 1) ? -score : score;
}

static int evaluate_raw_(const Pos &p, int fullmove) {
    bool chk = in_check_(p, p.stm);
    if (!has_legal_move(p)) return chk ? -30000 : 0;      // mate / stalemate
    if (side_insufficient(p, 0) && side_insufficient(p, 1)) return 0;
    return evaluate_raw_core(p, fullmove);
}

// ---------------------------------------------------------------------------
// Stage 1c: the quiescence search itself (evaluation.py:641 _quiescence).
// Captures+promotions alpha-beta, depth 2, MVV-LVA order, 200cp delta
// pruning. Returns raw centipawns — the tanh stays in Python so the value
// path never depends on which libm the extension was linked against.
//
// The root draw claim (can_claim_draw) stays in Python for v1: the caller
// computes it and passes `claimable`, and the check is applied at exactly
// the point in the sequence Python applies it.
//
// Move ordering: Python stable-sorts by -(victim - attacker + promo) with
// ties left in python-chess generation order; the native sort breaks ties by
// (from, to, promo) instead. Alpha-beta's fail-hard value is order-invariant
// over a fixed move set, but delta pruning at child nodes is window-dependent,
// so tie order COULD in principle shift a value — whether it ever does on
// real positions is exactly what the differential harness measures. If it
// reports mismatches, the fix is replicating python-chess's generation order,
// not weakening the gate.
// ---------------------------------------------------------------------------

static int qsearch(const Pos &p, int depth, int alpha, int beta, bool root,
                   bool claimable, int fullmove) {
    bool chk = in_check_(p, p.stm);
    std::vector<Move> legal = gen_legal(p);
    if (legal.empty()) return chk ? -30000 : 0;
    if (side_insufficient(p, 0) && side_insufficient(p, 1)) return 0;
    if (root && claimable) return 0;

    int stand_pat = evaluate_raw_core(p, fullmove);
    if (depth <= 0) return stand_pat;
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    struct Cand { int key; Move m; };
    std::vector<Cand> cands;
    cands.reserve(16);
    for (const Move &m : legal) {
        if (!is_qmove(p, m)) continue;
        // Victim from the target square only: an en-passant victim is not on
        // `to`, and Python's piece_at(to) -> None makes its value 0 there too.
        int victim_val = 0;
        if (p.occ[p.stm ^ 1] & bit(m.to))
            victim_val = PIECE_CP[piece_on(p, p.stm ^ 1, m.to)];
        if (stand_pat + victim_val + 200 < alpha && !m.promo) continue;
        int attacker_val = PIECE_CP[piece_on(p, p.stm, m.from)];
        int promo_val = (m.promo == 5) ? 800 : 0;      // queen promotions only
        cands.push_back({-(victim_val - attacker_val + promo_val), m});
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) {
        if (a.key != b.key) return a.key < b.key;
        if (a.m.from != b.m.from) return a.m.from < b.m.from;
        if (a.m.to != b.m.to) return a.m.to < b.m.to;
        return a.m.promo < b.m.promo;
    });

    for (const Cand &c : cands) {
        Pos child = make_move(p, c.m);
        // python-chess increments fullmove_number after Black's move.
        int child_fullmove = fullmove + (p.stm == 1 ? 1 : 0);
        int score = -qsearch(child, depth - 1, -beta, -alpha, false, false,
                             child_fullmove);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ---------------------------------------------------------------------------
// Python interface
// ---------------------------------------------------------------------------

static Pos unpack(u64 pawns, u64 knights, u64 bishops, u64 rooks, u64 queens,
                  u64 kings, u64 occ_w, u64 occ_b, int stm, int ep, u64 castling) {
    Pos p;
    u64 by_type[6] = {pawns, knights, bishops, rooks, queens, kings};
    for (int pt = 0; pt < 6; ++pt) {
        p.pc[0][pt] = by_type[pt] & occ_w;
        p.pc[1][pt] = by_type[pt] & occ_b;
    }
    p.occ[0] = occ_w;
    p.occ[1] = occ_b;
    p.all = occ_w | occ_b;
    p.stm = stm;
    p.ep = ep;
    p.castling = castling;
    return p;
}

using MoveTuple = std::tuple<int, int, int>;

static std::vector<MoveTuple> py_legal_moves(u64 pawns, u64 knights, u64 bishops,
        u64 rooks, u64 queens, u64 kings, u64 occ_w, u64 occ_b, int stm, int ep,
        u64 castling) {
    Pos p = unpack(pawns, knights, bishops, rooks, queens, kings,
                   occ_w, occ_b, stm, ep, castling);
    std::vector<MoveTuple> out;
    for (const Move &m : gen_legal(p))
        out.emplace_back(m.from, m.to, m.promo);
    return out;
}

static std::vector<MoveTuple> py_qmoves(u64 pawns, u64 knights, u64 bishops,
        u64 rooks, u64 queens, u64 kings, u64 occ_w, u64 occ_b, int stm, int ep,
        u64 castling) {
    Pos p = unpack(pawns, knights, bishops, rooks, queens, kings,
                   occ_w, occ_b, stm, ep, castling);
    std::vector<MoveTuple> out;
    for (const Move &m : gen_legal(p))
        if (is_qmove(p, m)) out.emplace_back(m.from, m.to, m.promo);
    return out;
}

static bool py_in_check(u64 pawns, u64 knights, u64 bishops, u64 rooks, u64 queens,
        u64 kings, u64 occ_w, u64 occ_b, int stm, int ep, u64 castling) {
    Pos p = unpack(pawns, knights, bishops, rooks, queens, kings,
                   occ_w, occ_b, stm, ep, castling);
    return in_check_(p, p.stm);
}

static bool py_has_legal(u64 pawns, u64 knights, u64 bishops, u64 rooks, u64 queens,
        u64 kings, u64 occ_w, u64 occ_b, int stm, int ep, u64 castling) {
    Pos p = unpack(pawns, knights, bishops, rooks, queens, kings,
                   occ_w, occ_b, stm, ep, castling);
    return has_legal_move(p);
}

static u64 perft(const Pos &p, int depth) {
    if (depth == 0) return 1;
    u64 nodes = 0;
    for (const Move &m : gen_legal(p)) {
        Pos n = make_move(p, m);
        nodes += (depth == 1) ? 1 : perft(n, depth - 1);
    }
    return nodes;
}

static u64 py_perft(u64 pawns, u64 knights, u64 bishops, u64 rooks, u64 queens,
        u64 kings, u64 occ_w, u64 occ_b, int stm, int ep, u64 castling, int depth) {
    Pos p = unpack(pawns, knights, bishops, rooks, queens, kings,
                   occ_w, occ_b, stm, ep, castling);
    return perft(p, depth);
}

static int py_evaluate_raw(u64 pawns, u64 knights, u64 bishops, u64 rooks,
        u64 queens, u64 kings, u64 occ_w, u64 occ_b, int stm, int ep,
        u64 castling, int fullmove) {
    Pos p = unpack(pawns, knights, bishops, rooks, queens, kings,
                   occ_w, occ_b, stm, ep, castling);
    return evaluate_raw_(p, fullmove);
}

static int py_qsearch(u64 pawns, u64 knights, u64 bishops, u64 rooks,
        u64 queens, u64 kings, u64 occ_w, u64 occ_b, int stm, int ep,
        u64 castling, int fullmove, int max_depth, bool claimable) {
    Pos p = unpack(pawns, knights, bishops, rooks, queens, kings,
                   occ_w, occ_b, stm, ep, castling);
    return qsearch(p, max_depth, -100000, 100000, true, claimable, fullmove);
}

PYBIND11_MODULE(chesskernel, m) {
    m.doc() = "Native leaf-evaluation kernel (stage 1: board + legal movegen)";
    m.def("legal_moves", &py_legal_moves);
    m.def("qmoves", &py_qmoves);
    m.def("in_check", &py_in_check);
    m.def("has_legal", &py_has_legal);
    m.def("perft", &py_perft);
    m.def("evaluate_raw", &py_evaluate_raw);
    m.def("qsearch", &py_qsearch);
}
