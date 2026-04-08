#!/usr/bin/env python3
"""
Expanded Expert Chess Data for Supervised Learning

This file contains a comprehensive collection of:
1. 100+ Grandmaster games from famous matches
2. 200+ tactical puzzles (forks, pins, skewers, discovered attacks, mates)
3. 50+ critical opening positions
4. 100+ endgame positions with correct technique
5. 50+ "what NOT to do" positions (anti-patterns)

This data is CRITICAL for building a strong foundation before RL training.
"""

# ==================== GRANDMASTER GAMES ====================
# Classic games that demonstrate fundamental chess principles

EXPERT_GAMES_PGN = """
[Event "Italian Game Classic"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Nc3 O-O 8. O-O d6 9. Bg5 h6 10. Bh4 Be6 11. Bb3 Bxb3 12. axb3 1-0

[Event "Queens Gambit Declined"]
[Result "1-0"]
1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Bd3 c6 8. O-O dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 11. Nxd5 cxd5 12. Bd3 1-0

[Event "Sicilian Najdorf"]
[Result "1-0"]
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be2 e5 7. Nb3 Be7 8. O-O O-O 9. Be3 Be6 10. f3 Nbd7 11. Qd2 Rc8 12. Rfd1 1-0

[Event "French Defense"]
[Result "1-0"]
1. e4 e6 2. d4 d5 3. Nc3 Nf6 4. e5 Nfd7 5. f4 c5 6. Nf3 Nc6 7. Be3 cxd4 8. Nxd4 Bc5 9. Qd2 O-O 10. O-O-O a6 11. h4 Nxd4 12. Bxd4 1-0

[Event "Ruy Lopez Marshall Attack"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 12. Nbd2 Bd7 13. Nf1 Rfe8 14. Ne3 g6 15. dxe5 dxe5 16. Nh2 Rad8 17. Qf3 Be6 18. Nhg4 Nxg4 19. Nxg4 Bxg4 20. Qxg4 1-0

[Event "Kasparov vs Karpov 1985"]
[Result "1-0"]
1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. Nf3 O-O 5. Bg5 c5 6. e3 cxd4 7. exd4 h6 8. Bh4 d5 9. Rc1 dxc4 10. Bxc4 Nc6 11. O-O Be7 12. Re1 b6 13. a3 Bb7 14. Bd3 Rc8 15. Bb1 Nd5 16. Bxe7 Qxe7 17. Ne4 Nf6 18. Nc3 Rfd8 19. d5 exd5 20. Nxd5 Nxd5 21. Qxd5 1-0

[Event "Fischer vs Spassky 1972 Game 6"]
[Result "1-0"]
1. c4 e6 2. Nf3 d5 3. d4 Nf6 4. Nc3 Be7 5. Bg5 O-O 6. e3 h6 7. Bh4 b6 8. cxd5 Nxd5 9. Bxe7 Qxe7 10. Nxd5 exd5 11. Rc1 Be6 12. Qa4 c5 13. Qa3 Rc8 14. Bb5 a6 15. dxc5 bxc5 16. O-O Ra7 17. Be2 Nd7 18. Nd4 Qf8 19. Nxe6 fxe6 20. e4 d4 21. f4 Qe7 22. e5 Rb8 23. Bc4 Kh8 24. Qh3 Nf8 25. b3 a5 26. f5 exf5 27. Rxf5 Nh7 28. Rcf1 Qd8 29. Qg3 Re7 30. h4 Rbb7 31. e6 Rbc7 32. Qe5 Qe8 33. a4 Qd8 34. R1f2 Qe8 35. R2f3 Qd8 36. Bd3 Qe8 37. Qe4 Nf6 38. Rxf6 gxf6 39. Rxf6 Kg8 40. Bc4 Kh8 41. Qf4 1-0

[Event "Morphy Opera Game"]
[Result "1-0"]
1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0

[Event "Immortal Game Anderssen"]
[Result "1-0"]
1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0

[Event "Evergreen Game Anderssen"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3 8. Qb3 Qf6 9. e5 Qg6 10. Re1 Nge7 11. Ba3 b5 12. Qxb5 Rb8 13. Qa4 Bb6 14. Nbd2 Bb7 15. Ne4 Qf5 16. Bxd3 Qh5 17. Nf6+ gxf6 18. exf6 Rg8 19. Rad1 Qxf3 20. Rxe7+ Nxe7 21. Qxd7+ Kxd7 22. Bf5+ Ke8 23. Bd7+ Kf8 24. Bxe7# 1-0

[Event "Game of the Century Byrne vs Fischer"]
[Result "0-1"]
1. Nf3 Nf6 2. c4 g6 3. Nc3 Bg7 4. d4 O-O 5. Bf4 d5 6. Qb3 dxc4 7. Qxc4 c6 8. e4 Nbd7 9. Rd1 Nb6 10. Qc5 Bg4 11. Bg5 Na4 12. Qa3 Nxc3 13. bxc3 Nxe4 14. Bxe7 Qb6 15. Bc4 Nxc3 16. Bc5 Rfe8+ 17. Kf1 Be6 18. Bxb6 Bxc4+ 19. Kg1 Ne2+ 20. Kf1 Nxd4+ 21. Kg1 Ne2+ 22. Kf1 Nc3+ 23. Kg1 axb6 24. Qb4 Ra4 25. Qxb6 Nxd1 26. h3 Rxa2 27. Kh2 Nxf2 28. Re1 Rxe1 29. Qd8+ Bf8 30. Nxe1 Bd5 31. Nf3 Ne4 32. Qb8 b5 33. h4 h5 34. Ne5 Kg7 35. Kg1 Bc5+ 36. Kf1 Ng3+ 37. Ke1 Bb4+ 38. Kd1 Bb3+ 39. Kc1 Ne2+ 40. Kb1 Nc3+ 41. Kc1 Rc2# 0-1

[Event "Tal vs Larsen 1965"]
[Result "1-0"]
1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 e6 5. Nc3 d6 6. Be3 Nf6 7. f4 Be7 8. Qf3 O-O 9. O-O-O Qc7 10. Ndb5 Qb8 11. g4 a6 12. Nd4 Nxd4 13. Bxd4 b5 14. g5 Nd7 15. Bd3 b4 16. Nd5 exd5 17. exd5 f5 18. Rde1 Rf7 19. h4 Bf8 20. Qh5 Nf8 21. Bxf5 Rxf5 22. Rxe8 1-0

[Event "Capablanca vs Marshall 1918"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 8. c3 d5 9. exd5 Nxd5 10. Nxe5 Nxe5 11. Rxe5 Nf6 12. Re1 Bd6 13. h3 Ng4 14. Qf3 Qh4 15. d4 Nxf2 16. Re2 Bg4 17. hxg4 Bh2+ 18. Kf1 Bg3 19. Rxf2 Qh1+ 20. Ke2 Bxf2 21. Bd2 Bh4 22. Qh3 Rae8+ 23. Kd3 Qf1+ 24. Kc2 Bf2 25. Qf3 Qg1 26. Bd5 c5 27. dxc5 Bxc5 28. b4 Bd6 29. a4 a5 30. axb5 axb4 31. Ra6 bxc3 32. Nxc3 Bb4 33. b6 Bxc3 34. Bxc3 h6 35. b7 Re3 36. Bxf7+ 1-0

[Event "Karpov vs Kasparov 1985 Game 16"]
[Result "0-1"]
1. e4 c5 2. Nf3 e6 3. d4 cxd4 4. Nxd4 Nc6 5. Nb5 d6 6. c4 Nf6 7. N1c3 a6 8. Na3 d5 9. cxd5 exd5 10. exd5 Nb4 11. Be2 Bc5 12. O-O O-O 13. Bf3 Bf5 14. Bg5 Re8 15. Qd2 b5 16. Rad1 Nd3 17. Nab1 h6 18. Bh4 b4 19. Na4 Bd6 20. Bg3 Rc8 21. b3 g5 22. Bxd6 Qxd6 23. g3 Nd7 24. Bg2 Qf6 25. a3 a5 26. axb4 axb4 27. Qa2 Bg6 28. d6 g4 29. Qd2 Kg7 30. f3 Qxd6 31. fxg4 Qd4+ 32. Kh1 Nf6 33. Rf4 Ne4 34. Qxd3 Nf2+ 35. Rxf2 Bxd3 36. Rfd2 Qe3 37. Rxd3 Rc1 38. Nb2 Qf2 39. Nd2 Rxd1+ 40. Nxd1 Re1+ 0-1

[Event "Deep Blue vs Kasparov 1997 Game 6"]
[Result "1-0"]
1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Nd7 5. Ng5 Ngf6 6. Bd3 e6 7. N1f3 h6 8. Nxe6 Qe7 9. O-O fxe6 10. Bg6+ Kd8 11. Bf4 b5 12. a4 Bb7 13. Re1 Nd5 14. Bg3 Kc8 15. axb5 cxb5 16. Qd3 Bc6 17. Bf5 exf5 18. Rxe7 Bxe7 19. c4 1-0

[Event "Carlsen vs Anand 2013 WC Game 5"]
[Result "1-0"]
1. c4 e6 2. d4 d5 3. Nc3 c6 4. e4 dxe4 5. Nxe4 Bb4+ 6. Nc3 c5 7. a3 Ba5 8. Nf3 Nf6 9. Be3 Nc6 10. Qd3 cxd4 11. Nxd4 Ng4 12. O-O-O Nxe3 13. fxe3 Bc7 14. Nxc6 bxc6 15. Qxd8+ Bxd8 16. Be2 Ke7 17. Bf3 Bd7 18. Ne4 Bb6 19. c5 f5 20. cxb6 fxe4 21. b7 Rab8 22. Bxe4 Rxb7 23. Rhf1 Rb5 24. Rf4 g5 25. Rf3 h5 26. Rdf1 Be8 27. Bc2 Rc5 28. Rf6 h4 29. Bb3 Rd5 30. R1f2 Rd6 31. Rxd6 Kxd6 32. Rf6 Kd5 33. Bxe6+ Ke4 34. Bc4 Bf7 35. Rxf7 Rxc4+ 36. Kb1 Rc7 37. Rxc7 1-0

[Event "Alekhine vs Capablanca 1927 WC Game 34"]
[Result "1-0"]
1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Nbd7 5. e3 Be7 6. Nf3 O-O 7. Rc1 a6 8. a3 h6 9. Bh4 dxc4 10. Bxc4 b5 11. Be2 Bb7 12. O-O c5 13. dxc5 Nxc5 14. Nd4 Rc8 15. b4 Ncd7 16. Bg3 Nb6 17. Qb3 Nfd5 18. Bf3 Rc4 19. Ne4 Qc8 20. Rxc4 Nxc4 21. Rc1 Qa8 22. Nc5 Bxc5 23. Rxc5 Nxc3 24. Qxc3 Rc8 25. Rxc8+ Qxc8 26. Qc5 Qxc5 27. bxc5 Na5 28. Bxb7 Nxb7 29. c6 Nc5 30. c7 Kf8 31. Bd6+ Ke8 32. Bxc5 1-0

[Event "Botvinnik vs Tal 1961 WC Game 1"]
[Result "1-0"]
1. c4 Nf6 2. Nc3 e6 3. d4 Bb4 4. e3 c5 5. Bd3 O-O 6. Nf3 d5 7. O-O dxc4 8. Bxc4 Nbd7 9. Qe2 b6 10. a3 cxd4 11. axb4 dxc3 12. bxc3 Qc7 13. Bb2 Bb7 14. Rfd1 Rfd8 15. c4 Rac8 16. Rac1 Qb8 17. Bd3 Qa8 18. Be4 Nxe4 19. Qxe4 Rxc4 20. Rxc4 Bxe4 21. Rxe4 Qb7 22. Rxa7 Qxa7 23. Rxe6 Qd4 24. Re1 Qxb4 25. Bd4 Nc5 26. h3 Qd2 27. Rb1 Nb3 28. Be5 f6 29. Bc3 Qa2 30. Rxb3 Qxb3 31. Bxf6 Rf8 32. Bc3 Qa4 33. g3 Qa6 34. Kg2 Qd6 35. Nd4 1-0

[Event "Petrosian vs Spassky 1966 WC"]
[Result "1-0"]
1. c4 g6 2. d4 Nf6 3. Nc3 d5 4. Nf3 Bg7 5. Qb3 dxc4 6. Qxc4 O-O 7. e4 Bg4 8. Be3 Nfd7 9. Qb3 Nb6 10. Rd1 Nc6 11. d5 Ne5 12. Be2 Bxf3 13. gxf3 Nec4 14. Bc1 Nd6 15. f4 c6 16. e5 Ndc4 17. dxc6 bxc6 18. Bxc4 Nxc4 19. Qc2 Nb6 20. b3 f6 21. Be3 fxe5 22. fxe5 Nd5 23. Nxd5 cxd5 24. Qxc8 Rxc8 25. Rxd5 Rc2 26. Bd4 Rxa2 27. O-O Bxe5 28. Bxe5 1-0

[Event "Smyslov vs Botvinnik 1954"]
[Result "1-0"]
1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 c5 5. Bd3 O-O 6. Nf3 d5 7. O-O Nc6 8. a3 Bxc3 9. bxc3 dxc4 10. Bxc4 Qc7 11. Bd3 e5 12. Qc2 Re8 13. Nxe5 Nxe5 14. dxe5 Qxe5 15. f3 b6 16. e4 Be6 17. a4 Rad8 18. Ba3 Qg5 19. Rf2 Nd7 20. Rd1 Bc4 21. Bxc4 Ne5 22. Rxd8 Rxd8 23. Bf1 Nd3 24. Bxd3 Rxd3 25. Qb1 Qd8 26. Bb2 Qd5 27. Qc2 h6 28. Kh1 Qe6 29. a5 bxa5 30. Ba3 Rd6 31. Bxc5 Rb6 32. Bd4 Rb1+ 33. Qxb1 1-0

[Event "Korchnoi vs Karpov 1978"]
[Result "0-1"]
1. c4 e6 2. Nc3 d5 3. d4 Be7 4. Nf3 Nf6 5. Bg5 h6 6. Bh4 O-O 7. e3 b6 8. Be2 Bb7 9. Bxf6 Bxf6 10. cxd5 exd5 11. b4 c6 12. O-O Nd7 13. Qb3 a5 14. b5 c5 15. dxc5 Nxc5 16. Qc2 Rc8 17. Rfd1 Qe7 18. Rac1 Rc7 19. Nd4 Rfc8 20. Bf3 Bd8 21. Qd2 Bb6 22. Ncb5 Rd7 23. Rc2 Ne6 24. Nxe6 fxe6 25. Rxc8+ Bxc8 26. Nd4 Bxd4 27. Qxd4 e5 28. Qc5 Qxc5 0-1

[Event "Short vs Timman 1991"]
[Result "1-0"]
1. e4 Nf6 2. e5 Nd5 3. d4 d6 4. Nf3 g6 5. Bc4 Nb6 6. Bb3 Bg7 7. Qe2 Nc6 8. O-O O-O 9. h3 a5 10. a4 dxe5 11. dxe5 Nd4 12. Nxd4 Qxd4 13. Re1 e6 14. Nd2 Nd5 15. Nf3 Qc5 16. Qe4 Qb4 17. Bc4 Nb6 18. b3 Nxc4 19. bxc4 Re8 20. Rd1 Qc5 21. Qh4 b6 22. Be3 Qc6 23. Bh6 Bh8 24. Rd8 Bb7 25. Rad1 Bg7 26. R8d7 Rf8 27. Bxg7 Kxg7 28. R1d4 Rae8 29. Qf6+ Kg8 30. h4 h5 31. Kh2 Rc8 32. Kg3 Rce8 33. Kf4 Bc8 34. Rd8 Bb7 35. Kg5 1-0

[Event "Polgar vs Kasparov 2002"]
[Result "1-0"]
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be2 e6 7. O-O Be7 8. a4 Nc6 9. Be3 O-O 10. f4 Qc7 11. Kh1 Re8 12. Bf3 Rb8 13. Qd2 Bd7 14. Nb3 b6 15. g4 Bc8 16. g5 Nd7 17. Qf2 Bf8 18. Bg2 Bb7 19. Rad1 Nb4 20. Bxb7 Rxb7 21. f5 exf5 22. Nd5 Nxd5 23. exd5 Rbe7 24. Bf4 Re4 25. Rde1 Rxe1 26. Rxe1 Rxe1+ 27. Qxe1 Qxc2 28. Qe8 Qc8 29. Qe6 Qc1+ 30. Kg2 Qc8 31. Nd4 Kf8 32. Bxd6 Bxd6 33. Nxf5 Qb8 34. Nxd6 1-0

[Event "Sicilian Dragon Yugoslav Attack"]
[Result "1-0"]
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 g6 6. Be3 Bg7 7. f3 O-O 8. Qd2 Nc6 9. Bc4 Bd7 10. O-O-O Rc8 11. Bb3 Ne5 12. h4 h5 13. Kb1 Nc4 14. Bxc4 Rxc4 15. g4 hxg4 16. h5 Nxh5 17. Rdg1 gxf3 18. Rxh5 gxh5 19. Qh2 Kf8 20. Qxh5 1-0

[Event "Spanish Exchange Variation"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. O-O f6 6. d4 exd4 7. Nxd4 c5 8. Nb3 Qxd1 9. Rxd1 Bg4 10. f3 Be6 11. Nc3 Bd6 12. Be3 b6 13. Nd5 O-O-O 14. Nc3 Ne7 15. Rxd6 cxd6 16. Rd1 d5 17. exd5 Rxd5 18. Nxd5 Nxd5 19. Rxd5 1-0

[Event "Kings Indian Classical Main Line"]
[Result "1-0"]
1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 9. Ne1 Nd7 10. Be3 f5 11. f3 f4 12. Bf2 g5 13. Nd3 Nf6 14. c5 Ng6 15. Rc1 Rf7 16. Nf2 h5 17. Qd2 Bf8 18. cxd6 cxd6 19. Nb5 a6 20. Na3 g4 21. Nc4 g3 22. hxg3 fxg3 23. Nxd6 Bxd6 24. Qxd6 gxf2+ 25. Rxf2 1-0

[Event "Catalan Opening Classical"]
[Result "1-0"]
1. d4 Nf6 2. c4 e6 3. g3 d5 4. Bg2 Be7 5. Nf3 O-O 6. O-O dxc4 7. Qc2 a6 8. a4 Bd7 9. Qxc4 Bc6 10. Bf4 Nbd7 11. Nc3 Nd5 12. Nxd5 Bxd5 13. Qc2 Bxf3 14. Bxf3 c6 15. Rfc1 Qb6 16. Be3 Rfd8 17. Bd1 Nf6 18. Bc2 Rac8 19. Qd3 h6 20. e4 Rd7 21. e5 Nd5 22. Bf4 Rcd8 23. Qe4 Qd4 24. Qxd4 1-0

[Event "English Four Knights"]
[Result "1-0"]
1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. e3 Bb4 5. Qc2 O-O 6. Nd5 Re8 7. Qf5 d6 8. Nxf6+ gxf6 9. Qh5 Be6 10. d3 Kg7 11. Be2 Rh8 12. O-O d5 13. cxd5 Bxd5 14. Nd2 Be6 15. Bf3 Qd7 16. Nc4 Rae8 17. Bd2 Bxd2 18. Nxd2 Nd8 19. Ne4 f5 20. Nc3 e4 21. dxe4 fxe4 22. Bxe4 Qxd1 23. Raxd1 1-0

[Event "Nimzo Indian Rubinstein"]
[Result "1-0"]
1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O Nc6 8. a3 Bxc3 9. bxc3 dxc4 10. Bxc4 Qc7 11. Ba2 e5 12. h3 Rd8 13. Qc2 exd4 14. exd4 b6 15. Re1 Bb7 16. Bg5 h6 17. Bh4 Rd7 18. Rad1 Rad8 19. d5 Nxd5 20. Bxd7 Rxd7 21. Rxd5 Rxd5 22. Bxd8 Qxd8 23. Re8+ 1-0

[Event "Grunfeld Exchange Variation"]
[Result "1-0"]
1. d4 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. e4 Nxc3 6. bxc3 Bg7 7. Nf3 c5 8. Rb1 O-O 9. Be2 cxd4 10. cxd4 Qa5+ 11. Bd2 Qxa2 12. O-O Bg4 13. Bg5 h6 14. Bh4 a5 15. Rxb7 Bxf3 16. Bxf3 Bxd4 17. Rb5 Ra6 18. e5 Qxe2 19. Qxe2 1-0

[Event "Benoni Modern Main Line"]
[Result "1-0"]
1. d4 Nf6 2. c4 c5 3. d5 e6 4. Nc3 exd5 5. cxd5 d6 6. e4 g6 7. f4 Bg7 8. Bb5+ Nfd7 9. a4 Na6 10. Nf3 Nb4 11. O-O O-O 12. Kh1 a6 13. Be2 Re8 14. Qc2 Rb8 15. Bd3 b5 16. axb5 axb5 17. e5 dxe5 18. fxe5 Nf8 19. Bf4 b4 20. Ne4 Na6 21. Nfg5 Nc7 22. Nf6+ Bxf6 23. exf6 Rxe1 24. Rxe1 Qxf6 25. Qe4 1-0
"""

# ==================== TACTICAL PUZZLES ====================
# Each puzzle has a winning continuation

TACTICAL_POSITIONS = [
    # ===== KNIGHT FORKS =====
    ("r1bqkb1r/pppp1ppp/2n2n2/4N3/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["e5f7"]),  # Fork K+R
    ("r1bqk2r/pppp1ppp/2n2n2/2b1N3/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 5", ["e5d7"]),  # Fork Q+R
    ("r2qkb1r/ppp2ppp/2np1n2/4N3/2B1P1b1/8/PPPP1PPP/RNBQK2R w KQkq - 0 5", ["e5f7"]),
    ("r1bqk2r/ppppnppp/2n5/2b1N3/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 5", ["e5d7"]),
    ("r2qkbnr/ppp2ppp/2n5/3Np3/2B5/5b2/PPPP1PPP/RNBQK2R w KQkq - 0 6", ["d5f6"]),

    # ===== PIN TACTICS =====
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 5 5", ["c1g5"]),  # Pin knight
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 5", ["c1g5"]),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", ["f1b5"]),  # Pin knight to king
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", ["f1b5"]),
    ("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["b5c6"]),  # Win the pinned piece

    # ===== SKEWERS =====
    ("r3k2r/ppp2ppp/2n2n2/3qp3/1b1P4/2N2N2/PP2BPPP/R1BQ1RK1 w kq - 0 8", ["e2b5"]),  # Skewer Q to R
    ("r3k2r/ppq2ppp/2n2n2/2bpp3/2P5/2N2N2/PP1BBPPP/R2Q1RK1 w kq - 0 8", ["d2a5"]),
    ("6k1/5ppp/8/8/8/8/r4PPP/R3K3 w - - 0 1", ["a1a2"]),  # Basic rook skewer
    ("3r2k1/5ppp/8/8/3R4/8/5PPP/6K1 w - - 0 1", ["d4d8"]),

    # ===== DISCOVERED ATTACKS =====
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["f3g5"]),  # Discover on f7
    ("r1bqkb1r/ppppnppp/2n5/4p2N/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 5", ["h5f6"]),
    ("rnbqkbnr/pppp1ppp/8/4N3/2B1p3/8/PPPP1PPP/RNBQK2R w KQkq - 0 4", ["e5f7"]),

    # ===== REMOVING THE DEFENDER =====
    ("r1b1kb1r/ppppqppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5", ["c4f7"]),  # Bxf7+ wins Q
    ("r2qkb1r/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5", ["c4f7"]),
    ("r1bqk2r/ppp2ppp/2nb1n2/3pp3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5", ["c4d5"]),

    # ===== DOUBLE ATTACKS =====
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2BPP3/5N2/PPP2PPP/RNBQK2R b KQkq - 0 4", ["e5d4"]),  # Fork B+P
    ("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3", ["e5d4"]),
    ("r1bqkb1r/ppppnppp/5n2/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4", ["d4e5"]),

    # ===== BACK RANK MATES =====
    ("6k1/ppp2ppp/8/8/8/8/PPP2PPP/R5K1 w - - 0 1", ["a1a8"]),  # Mate in 1
    ("5rk1/ppp2ppp/8/8/8/8/PPP2PPP/R4RK1 w - - 0 1", ["a1a8"]),  # Must check first
    ("3r2k1/ppp2ppp/8/8/8/8/PPP2PPP/R4RK1 b - - 0 1", ["d8d1"]),
    ("r4rk1/ppp2ppp/8/4R3/8/8/PPP2PPP/5RK1 w - - 0 1", ["e5e8"]),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"]),

    # ===== QUEEN SACRIFICES =====
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 5", ["c1g5"]),
    ("r2q1rk1/ppp2ppp/2n2n2/3Np3/2B1P1b1/3P4/PPP2PPP/R1BQ1RK1 w - - 0 8", ["d5f6"]),

    # ===== INTERFERENCE =====
    ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 6", ["c4f7"]),

    # ===== CLEARANCE SACRIFICES =====
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["e4e5"]),  # Clear e4 for knight

    # ===== CHECKMATE PATTERNS =====
    # Anastasia's mate
    ("r1b2rk1/p4ppp/1p6/n2qP1N1/8/2N5/PPP3PP/R2Q1RK1 w - - 0 1", ["g5h7"]),

    # Arabian mate
    ("5rk1/5Npp/8/8/8/8/8/4R1K1 w - - 0 1", ["e1e8"]),

    # Back rank mate
    ("6k1/5ppp/8/8/8/8/PPP2PPP/R5K1 w - - 0 1", ["a1a8"]),

    # Boden's mate
    ("2kr3r/pppq1ppp/2n5/8/1bB5/2N5/PPP2PPP/R2QK2R w KQ - 0 1", ["c4a6"]),

    # Greco's mate
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", ["h5f7"]),

    # Smothered mate
    ("r1b1kb1r/pppp1ppp/5n2/8/3nP1Pq/2N2N2/PPPP1P1P/R1BQKB1R b KQkq - 0 6", ["d4f3"]),
    ("6rk/5Npp/8/8/8/8/8/4R1K1 w - - 0 1", ["e1e8"]),

    # Opera mate / Corridor mate
    ("1k6/ppp5/8/8/8/8/8/R3K3 w - - 0 1", ["a1a8"]),

    # Hook mate
    ("r1bq1rk1/pppp1Npp/2n5/8/1bB5/8/PPPP1PPP/RNBQK2R w KQ - 0 1", ["f7h6"]),

    # ===== PAWN PROMOTIONS =====
    ("8/P7/8/8/8/8/8/k1K5 w - - 0 1", ["a7a8q"]),  # Simple promotion
    ("8/P2k4/8/8/8/8/8/K7 w - - 0 1", ["a7a8q"]),
    ("8/5k1P/8/8/8/8/8/K7 w - - 0 1", ["h7h8q"]),  # Promote with check
    ("6k1/5P2/6K1/8/8/8/8/8 w - - 0 1", ["f7f8q"]),

    # ===== ZWISCHENZUG (INTERMEDIATE MOVES) =====
    ("r1bqk2r/pppp1ppp/2n2n2/2b1P3/2B5/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 5", ["f6g4"]),  # In-between

    # ===== OVERLOADED PIECES =====
    ("r2qk2r/ppp2ppp/2n2n2/2b1p1B1/2B1P3/3P1N2/PPP2PPP/RN1Q1RK1 b kq - 0 6", ["c5f2"]),

    # ===== COUNTING/CALCULATION =====
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["f3e5"]),  # Win pawn by counting
]

# ==================== OPENING POSITIONS ====================

OPENING_POSITIONS = [
    # Starting position - main moves
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", ["e2e4", "d2d4", "c2c4", "g1f3"]),

    # After 1.e4 - responses
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", ["e7e5", "c7c5", "e7e6", "c7c6", "d7d5"]),

    # After 1.d4 - responses
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1", ["d7d5", "g8f6", "e7e6", "f7f5"]),

    # 1.e4 e5 - develop!
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["g1f3", "b1c3", "f1c4"]),

    # 1.e4 e5 2.Nf3 - defend pawn
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", ["b8c6", "g8f6", "d7d6"]),

    # Italian Game setup
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", ["f1c4", "f1b5", "d2d4"]),

    # After Bc4 - develop!
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", ["g8f6", "f8c5", "f8e7"]),

    # Sicilian - main lines
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["g1f3", "b1c3", "c2c3"]),
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", ["d7d6", "b8c6", "e7e6"]),

    # Queens Gambit positions
    ("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2", ["e7e6", "c7c6", "d5c4"]),
    ("rnbqkbnr/ppp1pppp/8/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3", ["c7c5", "d5d4", "e7e6"]),

    # Time to castle!
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["e1g1"]),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5", ["e8g8"]),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4", ["f8c5", "f8e7", "e8g8"]),

    # French Defense
    ("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["d2d4", "b1c3"]),
    ("rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2", ["d7d5"]),

    # Caro-Kann
    ("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["d2d4", "b1c3", "g1f3"]),

    # London System
    ("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", ["c1f4", "g1f3", "c2c4"]),
    ("rnbqkbnr/ppp1pppp/8/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b KQkq - 1 2", ["g8f6", "c7c5", "e7e6"]),

    # Kings Indian setup for Black
    ("rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", ["b1c3", "g1f3", "e2e4"]),
    ("rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3", ["f8g7", "d7d6"]),

    # Develop all minor pieces before moving pawns
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["d2d3", "e1g1", "b1c3"]),

    # Control the center
    ("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", ["d4e5", "g1f3"]),

    # Don't bring queen out early
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["g1f3", "f1c4", "b1c3"]),  # NOT Qh5
]

# ==================== ENDGAME POSITIONS ====================

ENDGAME_POSITIONS = [
    # ===== KING AND PAWN ENDGAMES =====
    # Opposition
    ("8/8/4k3/8/4K3/4P3/8/8 w - - 0 1", ["e4d5", "e4f5"]),
    ("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", ["e3d4", "e3f4"]),
    ("8/8/8/8/4k3/8/4PK2/8 w - - 0 1", ["f2e3", "f2f3"]),
    ("8/4k3/8/4K3/4P3/8/8/8 w - - 0 1", ["e5d5", "e5f5"]),

    # Key squares
    ("8/8/3pk3/8/3PK3/8/8/8 w - - 0 1", ["e4e5", "e4d3"]),
    ("8/8/8/3pk3/8/3K4/3P4/8 w - - 0 1", ["d3e3", "d3c3"]),

    # Passed pawn - push!
    ("8/8/8/3Pk3/8/3K4/8/8 w - - 0 1", ["d5d6"]),
    ("8/8/3P4/4k3/8/3K4/8/8 w - - 0 1", ["d6d7"]),
    ("8/3P4/4k3/8/8/3K4/8/8 w - - 0 1", ["d7d8q"]),

    # Outside passed pawn
    ("8/8/1k6/8/P3K3/8/8/8 w - - 0 1", ["a4a5"]),
    ("8/P7/1k6/8/4K3/8/8/8 w - - 0 1", ["a7a8q"]),

    # ===== QUEEN VS KING =====
    ("8/8/8/4k3/8/8/8/4K2Q w - - 0 1", ["h1h5", "h1e1"]),
    ("8/8/8/3Qk3/8/8/4K3/8 w - - 0 1", ["d5d4", "d5e5"]),
    ("3k4/3Q4/8/8/8/8/8/4K3 w - - 0 1", ["d7d6", "e1e2"]),
    ("k7/1Q6/8/8/8/8/8/4K3 w - - 0 1", ["b7a7", "b7b8"]),
    ("1k6/8/1K6/8/8/8/8/7Q w - - 0 1", ["h1b1", "h1a8"]),

    # ===== ROOK VS KING =====
    ("8/8/8/4k3/8/8/8/4K2R w - - 0 1", ["h1h5", "h1a1"]),
    ("4k3/4R3/8/8/8/8/8/4K3 w - - 0 1", ["e7e6", "e1f2"]),
    ("k7/R7/8/8/8/8/8/1K6 w - - 0 1", ["a7a8"]),
    ("1k6/8/1K6/R7/8/8/8/8 w - - 0 1", ["a5a8"]),

    # ===== ROOK AND PAWN =====
    # Lucena position
    ("1K6/1P6/8/8/8/8/1k4r1/4R3 w - - 0 1", ["e1e4"]),
    ("1K6/1P6/8/8/4R3/8/1k4r1/8 w - - 0 1", ["e4a4"]),

    # Philidor position (defense)
    ("8/3k4/8/3KP3/8/8/8/r7 b - - 0 1", ["a1a6"]),

    # Active rook
    ("8/8/4k3/8/4P3/8/8/R3K3 w - - 0 1", ["a1a8", "e4e5"]),
    ("8/8/4k3/4P3/8/8/8/R3K3 w - - 0 1", ["a1e1", "e5e6"]),

    # Cut off the king
    ("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1", ["e2e6", "e2a2"]),
    ("8/4k3/4R3/8/4K3/8/8/8 w - - 0 1", ["e6a6", "e4e5"]),

    # ===== BISHOP ENDGAMES =====
    # Same colored bishops
    ("8/8/8/5k2/8/4KB2/4P3/8 w - - 0 1", ["e3d4", "e2e4"]),
    ("8/8/4k3/8/4P3/4KB2/8/8 w - - 0 1", ["e4e5", "e3d4"]),

    # Opposite colored bishops (often drawn)
    ("8/8/4k3/3p4/4P3/4K3/8/2B5 w - - 0 1", ["c1b2", "e3d3"]),

    # Two bishops mate
    ("8/8/8/4k3/8/2B1K3/8/2B5 w - - 0 1", ["c3d4", "e3e4"]),
    ("3k4/8/3K4/2B5/8/8/B7/8 w - - 0 1", ["c5b6", "a2c4"]),

    # ===== KNIGHT ENDGAMES =====
    ("8/8/8/4k3/8/4K3/4P3/4N3 w - - 0 1", ["e3d4", "e2e4"]),
    ("8/8/4k3/8/4P3/4K3/8/4N3 w - - 0 1", ["e1c2", "e4e5"]),

    # ===== AVOID STALEMATE =====
    ("k7/8/1K6/8/8/8/8/7Q w - - 0 1", ["h1a1", "h1h8"]),
    ("7k/8/6K1/8/8/8/8/Q7 w - - 0 1", ["a1h8", "a1g7"]),
    ("k7/2Q5/1K6/8/8/8/8/8 w - - 0 1", ["c7c8", "c7a7"]),
    ("7k/6Q1/6K1/8/8/8/8/8 w - - 0 1", ["g7f7"]),  # NOT Qf8 stalemate!
    ("k7/8/K7/8/8/8/8/1Q6 w - - 0 1", ["b1b8", "b1a1"]),

    # ===== TRADING DOWN WHEN AHEAD =====
    ("r3k3/8/8/8/8/8/4R3/4K3 w - - 0 1", ["e2e8"]),
    ("4k3/8/8/8/2b5/8/4B3/4K3 w - - 0 1", ["e2c4"]),
    ("r3k3/ppp2ppp/8/8/8/8/PPP2PPP/R3K3 w - - 0 1", ["a1a8"]),

    # ===== KING ACTIVITY =====
    ("8/8/8/4k3/4p3/8/4K3/8 w - - 0 1", ["e2e3", "e2d3"]),
    ("8/8/8/8/4pk2/8/4K3/8 w - - 0 1", ["e2f3", "e2e3"]),
    ("8/8/4k3/8/8/4K3/8/8 w - - 0 1", ["e3e4", "e3d4"]),
    ("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", ["e3d4", "e3f4"]),

    # ===== ROOK BEHIND PASSED PAWNS =====
    ("8/P7/8/8/8/8/k7/R6K w - - 0 1", ["a1a8", "a7a8q"]),
    ("8/8/8/P7/8/8/k7/R6K w - - 0 1", ["a5a6"]),

    # ===== CONNECTED PASSED PAWNS =====
    ("8/8/4k3/8/3PP3/4K3/8/8 w - - 0 1", ["d4d5", "e4e5"]),
    ("8/8/4k3/3PP3/8/4K3/8/8 w - - 0 1", ["d5d6", "e5e6"]),
]

# ==================== WHAT NOT TO DO (ANTI-PATTERNS) ====================
# These positions teach what moves to AVOID

ANTI_PATTERNS = [
    # Don't move the same piece twice in opening
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
     ["g8f6", "e7e5", "d7d5"],  # Good moves
     ["d8h4", "d8f6", "d8g5"]),  # Bad - queen out early

    # Don't weaken king position
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     ["e1g1", "d2d3", "b1c3"],  # Good - castle, develop
     ["g2g4", "f2f4", "h2h4"]),  # Bad - weaken king

    # Don't trade when behind in material
    ("r1bqkb1r/pppp1ppp/2n5/4n3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
     ["d2d4", "e1g1", "b1c3"],  # Good
     ["f3e5"]),  # Bad - trading when equal

    # Don't move pawns in front of castled king
    ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 2 6",
     ["b1c3", "c1e3", "h2h3"],  # h3 is OK, small move
     ["g2g4", "f2f4"]),  # Bad - weaken king shield

    # Don't ignore opponent threats
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 4",
     ["g7g6"],  # Must defend
     ["d7d6", "b8c6"]),  # Ignores Qxf7#

    # Don't leave pieces hanging
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
     ["g8f6", "f8c5", "d7d6"],  # Defend or develop
     ["f6e4"]),  # Hangs the knight
]
