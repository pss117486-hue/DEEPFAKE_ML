APPEND (or merge) THIS into the "## 5. Known limitations / honest caveats"
section of your existing README.md -- add it as a new bullet alongside the
existing ones (face-only coverage, generalization gap, yolov8_face note,
uncertain-band note):

- **GAN/diffusion whole-face synthesis detection is disabled, not just
  weak.** A fifth branch was added to catch StyleGAN/diffusion-generated
  faces (a different artifact family from the face-swap detection the
  other branches do). Three training attempts -- raw RGB input; harmonized
  data + heavy augmentation; a fixed SRM noise-residual front end -- all
  hit near-perfect AUC (0.99+) on data from the same source(s) they
  trained on, but only 0.53-0.58 AUC (essentially chance) against
  genuinely independent StyleGAN/diffusion holdouts the model never saw
  during training. In other words: it learned to recognize its own
  training data, not synthesis artifacts that generalize. Rather than
  ship a branch that looks confident but isn't, it's disabled via
  `config.GAN_BRANCH_ENABLED = False` -- the code and checkpoint exist for
  future work, but don't rely on it. Closing this gap will need
  substantially more diverse real+fake training sources (many camera/
  photo sources, many generator families) rather than another
  architecture tweak; re-enable only after re-validating against a
  holdout that shares no source with training data, not just a better
  same-source val_auc.
