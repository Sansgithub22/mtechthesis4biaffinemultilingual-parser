# Presentation Pitch — Speaking Notes (Slide by Slide)

Simple-language script for pitching the thesis. Tone: confident, conversational, don't read bullets verbatim. The narrative is: **Part 1 — my novel alignment losses beat naive fine-tuning. Part 2 — my UD-Bridge beats zero-shot on the real benchmark.**

---

## Slide 1 — Title

"Good morning, sir. My thesis is on building a dependency parser for Bhojpuri — a low-resource language. The idea is to transfer from Hindi using parallel bottleneck adapters and syntax-aware alignment. I'll walk through the work in two parts."

---

## Slide 2 — Outline

"The talk is structured in two parts. Part 1 is about my novel cross-lingual alignment architecture, developed on the Hindi-Bhojpuri aligned data. Part 2 is about applying the lessons to the real Bhojpuri benchmark — the BHTB treebank. Part 1 builds the architecture, and Part 2 evaluates on the actual benchmark."

---

## Slide 3 — Dependency Parsing in One Slide

"Just to set the stage — dependency parsing is about finding the grammatical structure of a sentence. Every word points to exactly one head, and each arc has one label like subject or object. In 'Ram ghar jaa rahaa hai', 'Ram' is the subject of 'jaa' and 'ghar' is where he's going. It feeds into machine translation, QA, and information extraction. I measure accuracy with UAS — correct heads — and LAS — correct heads and labels."

---

## Slide 4 — Why Bhojpuri?

"Bhojpuri has 50 to 60 million native speakers — that's huge — but almost zero NLP tools exist for it. It was historically mislabelled as a Hindi dialect and only got its own ISO code in 2006. The challenge is there's no large annotated Bhojpuri treebank, so we need to transfer from Hindi. Fortunately Bhojpuri is linguistically very close to Hindi — same word order, shared vocabulary, same Devanagari script. To work with, I have a Hindi-Bhojpuri aligned corpus of about 30,000 pairs and the Hindi HDTB treebank with 13,000 sentences."

---

## Slide 5 — Building Blocks

"The technical foundation briefly. The backbone is XLM-RoBERTa, pretrained on 100 languages including both Hindi and Bhojpuri. I use the Trankit toolkit which supports language-specific adapters. Pfeiffer adapters are small bottleneck modules — around 100K parameters per language — so I can freeze the backbone and avoid catastrophic forgetting. And the biaffine scorer is the standard way to score head-dependent pairs."

---

# PART 1 — Novel Cross-Lingual Alignment

---

## Slide 6 — System F — Warm-Start Fine-Tuning (Baseline)

"System F is the straightforward first attempt. I start from a Hindi-pretrained Trankit checkpoint and fine-tune all the parameters on the Hindi-Bhojpuri aligned data. Standard cross-entropy training. On the Dev set I get only 27.57% UAS and 17.75% LAS — quite weak. Full fine-tuning on a small aligned dataset without explicit cross-lingual alignment just doesn't transfer well. The research question becomes: can we do much better by explicitly aligning Hindi and Bhojpuri representations?"

---

## Slide 7 — System G — Parallel Adapters + MSE Alignment

"System G is my first novel contribution, with two design changes. First, I freeze XLM-R completely and only train small adapters plus the heads — about 200K parameters total, a thousand times fewer than F. Second, I add an MSE loss that pulls Bhojpuri hidden states toward Hindi hidden states at matched positions. So Hindi representations become a target that Bhojpuri mimics. Final Dev result: 50.02% LAS — beats F, with no forgetting."

---

## Slide 8 — System H — SACT: Syntax-Aware Cross-Lingual Transfer

"System H is my main architectural contribution. The problem with MSE is it's blunt — it aligns every representation uniformly, even function words and punctuation which don't carry structural information. SACT replaces MSE with three syntax-aware losses. First, content-word cosine alignment — nouns and verbs get weight 1.5, function words 0.3. Second, Arc-KL distillation — the Hindi parser is a teacher and I transfer its parsing decisions using KL divergence on arc scores, not just features. Third, Cross-lingual Tree Supervision — I reuse clean Hindi gold heads directly for matched Bhojpuri tokens. Result: 50.08% LAS on Dev — my best architecture on aligned data."

---

## Slide 9 — Results on Aligned Dev Set

"Putting Part 1 together. F baseline is only 17.75% LAS. G with MSE jumps to 50.02% — that's a huge plus 32 point gain with a thousand times fewer trainable parameters, because the frozen backbone plus explicit alignment fixes what naive fine-tuning breaks. H with SACT gets 50.08% at convergence — only 0.06 points better than G at the end, but at epoch 1 it's already plus 3.3 points ahead — SACT converges dramatically faster. Both my novel designs massively beat the naive fine-tuning baseline."

---

## Slide 10 — Learning Curves G vs H

"This is where SACT really shines. System H starts 3.3 LAS points ahead of G at epoch 1 and almost plateaus by epoch 2. G takes 10 epochs to catch up. The syntax-aware losses give a much stronger gradient signal from the beginning — practically valuable when compute or time is limited."

---

## Slide 11 — Ablation

"I ran ablations to confirm every SACT component pulls its weight. Removing any single loss drops performance. Hindi regularisation contributes 0.46 points, cosine alignment 0.37, CTS 0.21, and Arc-KL 0.14. Removing all auxiliary losses drops 1.15 points. None of the components is redundant."

---

## Slide 12 — Part 1 Takeaway

"So the Part 1 story is simple — my novel cross-lingual alignment losses improve over naive fine-tuning on the Hindi-Bhojpuri aligned corpus. Freezing the backbone beats full fine-tuning. Syntax-aware supervision beats uniform MSE. The progression from F to G to H is a clean architectural story. But aligned-data performance is only one side of the coin — the real benchmark for Bhojpuri lies elsewhere, which is what Part 2 is about."

---

# PART 2 — Real-World Benchmark Evaluation

---

## Slide 13 — The Bhojpuri Treebank (BHTB) — The Real Benchmark

"The actual evaluation target for any Bhojpuri parser is BHTB — the only gold treebank for Bhojpuri. It has 357 manually annotated sentences in strict Universal Dependencies schema. It's test-only, so we can't train on it. Now the natural question is: can I just evaluate my Part 1 systems F, G, H on BHTB? The answer is no — and here's why. The aligned corpus uses auto-transferred labels, which have multiple roots, cycles, and non-UD relations. BHTB uses strict UD. So F, G, and H learn the 'aligned-data dialect', not the UD dialect BHTB uses. The training signal lives in the wrong space. To beat BHTB, I need a completely different strategy — train in the UD label space."

---

## Slide 14 — System A — Zero-Shot Baseline on BHTB

"System A is the baseline for the UD benchmark. I train Trankit only on Hindi HDTB — 13,000 UD-labelled sentences — with zero Bhojpuri training. Then apply it directly to Bhojpuri. The only bridge is XLM-R's shared multilingual representations. Because HDTB is native UD, labels automatically live in the right schema. Result: 52.78% UAS and 35.36% LAS on BHTB. A reasonable score with no labelled Bhojpuri data at all. This is the number to beat."

---

## Slide 15 — System K — UD-Bridge via Silver Self-Training

"System K is my novel contribution for the benchmark. The premise: schema consistency beats data quantity. Three steps. Step 1, take the same 30,966 Bhojpuri sentences I used in Part 1 — from the Hindi-Bhojpuri aligned corpus — strip off the auto-transferred labels, and run System A on the raw tokens to generate fresh UD-style silver labels. Step 2, concatenate the silver Bhojpuri data with HDTB gold. Step 3, train a new Trankit model on this combined corpus. The whole training signal now lives in UD space, so gains transfer directly to BHTB. Result: 54.27% UAS and 36.70% LAS on BHTB — beating System A by +1.49 UAS and +1.34 LAS."

---

## Slide 16 — Why UD-Bridge Works

"The schema-consistency argument. System A already parses in UD space, so its silver Bhojpuri labels are noisy but schema-consistent. Gold HDTB adds correct UD structure and regularises training. The silver errors are random, not systematic — they're not misaligned with the UD schema the way the auto-transferred labels were. And any gains on the training objective transfer directly to BHTB because both are in the same annotation space. The core claim: noisy data in the right annotation space beats high-quality data in the wrong annotation space."

---

## Slide 17 — Results on BHTB Gold

"System A sits at 52.78 UAS and 35.36 LAS. System K reaches 54.27 UAS and 36.70 LAS — a clean win of +1.49 UAS and +1.34 LAS. K adds Bhojpuri-specific supervision that A never saw, keeps everything in UD schema, and HDTB gold prevents silver noise from dominating. This is state-of-the-art on real Bhojpuri parsing."

---

## Slide 18 — Error Analysis

"Breaking down System A's errors by relation type. Local structural relations transfer very well — case markers at 83%, root at 50%, punct at 46%. But clausal relations collapse — xcomp and ccomp at 0%, acl at 1%, advcl at 4%. The pattern: short-arc morphologically-marked relations transfer from Hindi, but clause-level relations — where Bhojpuri diverges syntactically — need native Bhojpuri supervision. This also tells us where future gold data should focus."

---

## Slide 19 — Part 2 Takeaway

"Part 2 story: UD-Bridge self-training beats the zero-shot baseline on BHTB — advancing the state of the art on real Bhojpuri parsing. What made the difference is schema alignment between training and test, plus Bhojpuri-specific supervision from silver data, anchored by HDTB gold. The methodological lesson is bigger than just this system: schema consistency matters as much as model quality, and self-training is a clean bridge into low-resource UD settings."

---

# CONCLUSION

---

## Slide 20 — Key Findings Across Both Parts

"Four main findings across both parts. One — freezing the backbone is essential, otherwise full fine-tuning forgets the multilingual prior. Two — syntax-aware alignment beats representation-level alignment, because it supervises decisions directly. Three — schema consistency matters as much as model quality. Strong performance on one dataset doesn't automatically transfer to a differently-labelled benchmark. Four — self-training with schema-consistent silver data is a clean way to bridge low-resource settings."

---

## Slide 21 — Contributions

"My contributions split naturally into two groups. From Part 1, two architectural contributions: System G — frozen XLM-R with parallel adapters and MSE alignment, and System H — SACT, my syntax-aware alignment framework achieving 50.08% LAS on Dev. From Part 2, one benchmark contribution: System K — UD-Bridge, which targets state-of-the-art on BHTB through silver self-training. And across both, one methodological insight — I identified and characterised the annotation-schema mismatch problem and showed that noisy but schema-consistent supervision outperforms clean but schema-mismatched supervision."

---

## Slide 22 — Future Work

"Short term: complete K on BHTB and try iterative silver-to-model-to-silver self-training. Data augmentation: synthetic UD treebanks through CPG-to-UD conversion plus Hindi-to-Bhojpuri translation, and extending BHTB gold annotations. Architecture: multi-layer adapters across all XLM-R layers, non-projective MST decoding, and morphology-aware alignment. Generalisation: the framework should extend to Maithili, Magahi, Awadhi, and Rajasthani — related languages facing similar low-resource problems."

---

## Slide 23 — Thank You

"That's the main talk. Thank you, sir — happy to take any questions."

---

## Backup Slides (24–26)

If asked, say:

- **Hyperparameters:** "Adapter bottleneck is 64, standard Pfeiffer setup. Arc MLP dim 500, label MLP 100, AdamW with learning rate 2e-3 for adapters and 5e-5 for full fine-tuning. Max 60 epochs, batch size 16."
- **Datasets:** "HDTB is 13,304 sentences, the Hindi-Bhojpuri aligned corpus is 30,966 pairs, the Dev split has 3,097 sentences, and BHTB is 357 test sentences. The key asymmetry is that only 357 gold Bhojpuri sentences exist — that's why transfer and self-training are essential."
- **SACT loss details:** "Full weights — Bhojpuri loss 1.0, Hindi auxiliary 0.5, cosine 0.4, Arc-KL 2.0, CTS 0.2. Arc-KL dominates because distillation gives the most direct supervision on parsing decisions."

---

## General Tips While Presenting

- Don't read bullets — speak the idea behind them.
- The **Part 1 / Part 2 split** is the spine of the talk. Never collapse them together — finish the first story before starting the second.
- The **bridge slide (Slide 12, Part 1 Takeaway)** is where you close Part 1 cleanly and hint that Part 2 is coming. Pause here for a beat.
- On **Slide 13 (BHTB intro)**, the schema mismatch explanation is critical — slow down and walk through why F/G/H can't just be applied to BHTB.
- **SACT slide (8)** and **UD-Bridge slide (15)** are the two novel-contribution highlights. Explain each patiently.
- On results slides, lead with the headline number, then the caveat.
- If the supervisor interrupts with a question, answer directly and don't rush back to the bullets.
