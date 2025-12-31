# Kannada ASR Dataset Compilation

## Primary Datasets

| Dataset Title | Link | Hours (KA) | Description & Notes |
| :--- | :--- | :--- | :--- |
| **Shrutilipi** (Hugging Face / AI4Bharat) | [Dataset Link](https://huggingface.co/datasets/ai4bharat/Shrutilipi) | **460** | Professional, formal news-reading style. Clean audio. |
| **Kathbath** (Hugging Face / AI4Bharat) | [Dataset Link](https://huggingface.co/datasets/ai4bharat/kathbath) | **150** | High quality, verified transcriptions. Smaller in size compared to Shrutilipi. |
| **IndicVoices** (Hugging Face / AI4Bharat) | [Dataset Link](https://huggingface.co/datasets/ai4bharat/IndicVoices) | **73** | A mix of read text, extempore (just talking), and conversational speech. |
| **Vaani** (ARTPARK / IISc) | [Dataset Link](https://huggingface.co/datasets/ARTPARK-IISc/Vaani) | **160** | Diverse dialects, strong rural speech coverage.<br>**Note:** Actually 2600 hours but only 160 is transcribed. Contains English words inserted between tags `<>`. |
| **IISc MILE** (Kannada–Tamil Read Speech) | *Search: IISc-MILE Kannada ASR Corpus* | **~350** | Clean, read speech in controlled settings. |
| **SPRING-INX** (IIT Madras) | [Dataset Link](https://asr.iitm.ac.in/dataset) | **~97** | High-quality academic transcriptions across languages. |
| **ReSPIN / RESPIN-S1.0** (IISc SPIRE Lab) | [Home Page](https://spiredatasets.ee.iisc.ac.in/respincorpus)<br>[Documentation](https://docs.google.com/document/d/1Ef7_NS3BqIeXe4BuB_mfq8RJramr5eSGLIvtMgbf7O8/edit?tab=t.0) | **~1,000** | Extensive dialect-focused Kannada speech. |
| **OpenSLR 80** | [Dataset Link](https://www.openslr.org/79/) | **~30** | Crowdsourced high-quality Kannada multi-speaker speech data set. |
| **Bhashini / ULCA** (Govt. of India) | [Portal Link](https://bhashini.gov.in/ulca) | *Var* | Government-aggregated Indian language datasets. |

---

## Excluded / Not Suitable Datasets

| Dataset Title | Link | Reason for Exclusion |
| :--- | :--- | :--- |
| **Svarah** | [Dataset Link](https://huggingface.co/datasets/ai4bharat/Svarah) | **English Only:** This is an English ASR dataset (Indian dialects), so it is not suitable for Kannada. |
| **Vistaar** | [GitHub Link](https://github.com/AI4Bharat/vistaar) | **Redundant:** Not required; it is mainly a composition of Shrutilipi + Kathbath + NPTEL (often used for bad audio quality benchmarks). |
| **MUCS** (Multilingual & Code-Switching) | [Link 1](https://www.openslr.org/103/) / [Link 2](https://www.openslr.org/104/) | **No Kannada:** Mixed-language speech, but does not contain Kannada data. |
| **SYSPIN** | [Dataset Link](https://spiredatasets.ee.iisc.ac.in/syspincorpus) | **TTS Focused:** Very clear audio meant for Text-to-Speech; generally not suitable for robust ASR training. |