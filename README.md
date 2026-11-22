# No JavaScript, Just Vibes

This README was written by me, but the code was written by Antigravity and Gemini Carbon.

A collection of vibe-coded HTML+CSS only web demos that use **zero** JavaScript.

**Live Demos:** [https://dcthetall.github.io/nojs/](https://dcthetall.github.io/nojs/)

## Projects

All projects include Python code which generate the relevant HTML and CSS.

### [AES](https://dcthetall.github.io/nojs/aes/) [`aes/`](aes/)

- Full 10-round 128-bit AES encryption
- Can use arbitrary 128-bit input
- Uses arbitrary 8-bit vector or NIST FIPS-197 Appendix B vector as keys

### [Lisp Interpreter](https://dcthetall.github.io/nojs/lisp/) [`lisp/`](lisp/)

- Live S-expression lexing
- Parenthesis matching
- Lexical and mutable environment
- Supports higher order and recursive functions 

### [MNIST Convolutional Neural Net](https://dcthetall.github.io/nojs/cnn/) [`cnn/`](cnn/)

- Uses real CNN trained on MNIST dataset
- GPU acceleration using HTML5 SVG
- Over 95% accuracy in bit-for-bit simulation on MNIST test set
- Renders latent activations using interactable UI

### [Chess](https://dcthetall.github.io/nojs/chess/) [`chess/`](chess/)

- 1-player mode against an HTML5 SVG `<filter>` 2-ply + Ply-3 Quiescence Minimax AI
- 2-player local Pass & Play
- Stockfish-distilled 2,623-parameter 8-bit quantized CNN Value Network evaluating 1024 FIDE-legal leaves in parallel with exact Ply-1/Ply-2 checkmate detection & SEE-filtered Ply-3 Quiescence
- Full FIDE chess rules in CSS
- Supports undo, board flip, live neural value/latent heatmaps, and preset tactical positions
