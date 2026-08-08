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

#include <cstdint>
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

PYBIND11_MODULE(chesskernel, m) {
    m.doc() = "Native leaf-evaluation kernel (stage 1: board + legal movegen)";
    m.def("legal_moves", &py_legal_moves);
    m.def("qmoves", &py_qmoves);
    m.def("in_check", &py_in_check);
    m.def("has_legal", &py_has_legal);
    m.def("perft", &py_perft);
}
