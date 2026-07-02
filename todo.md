- [ ] Fine-tuning 
  - [ ] Single model 
  - [ ] Ensemble
  
- [ ] "Zero-shot" E-DAIC test (Train on DAIC-WOZ, test on E-DAIC)
  - [ ] No retraining/threshold tuning 
  - [ ] **Verify carefully that E-DAIC uses compatible labels (PHQ-based depression labels or a mapping that is comparable to DAIC-WOZ binary task)**

- [x] Update README (rewritten as TC-MIL entry point; grounded results + real paths)

- [ ] Questions for professor 
  - [x] Paper: is there a page limit? 
  - [ ] How deep should the analysis of "failed" iterations/ablations be? Should it be like a real conference paper? 
  - [x] Baselines comparison: some results are suspicious, should we trust them or reimplement them under our protocol?
    - [ ] Should we address it in the paper or undersell our work?
  - [x] Does the presentation have a time-limit? 
  - [ ] Should we include multimodal baselines or stick to text-only?
  - [ ] Should we train model on E-DAIC or try model trained on DAIC-WOZ zero-shot on E-DAIC to check if it actually learned some depression signals or if just overfitted the DAIC-WOZ dataset? Because if we train it on E-DAIC it is reasonable to think it will work and it won't be very informative (+ a lot of work/experiments to be on par with the DAIC-WOZ experiments)
  - [ ] 

- [ ] Include validation-only baselines
- [ ] Compact paper 
  - [ ] Remove significance tables (almost all p are significant, few lines in prose will do it)
  - [ ] Remove dialogue mean and flat MIL baselines from tables (less interesting/informative)
  - [ ] Shorter description of our own prior baselines (DAMIL-R, SS-DAMIL-R)

- Page limit: 10 (check if we can have some more)
- Baselines comparison: take results at face value OK, add baselines with results on validation set 
- Presentation time limit: 10 minutes 
  - Presentation impacts final grade (what you put in the slide, how you present)
  - Should be clear, understandable, easy to follow for a technical audience who is not up-to-date with the history of the project 
  - Think of it as a presentation to the manager who commissioned you the project, and you have to "convince" them to pay you (convince them of the goodness of your work)

