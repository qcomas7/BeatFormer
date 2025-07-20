# BeatFormer

To train BeatFormer-SCL, follow the next steps:

1) **Download the PURE dataset**:  
   - Stricker, R., Müller, S., Gross, H.-M. "Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot" in *Proc. 23rd IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014)*, Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014.

2) **Install the required packages**:  
   Install the necessary Python packages using `requirements.txt` by running the following command:
   
   ```bash
   pip install -r requirements.txt


3) **Preprocess the data**:
	
	Run preprocess/Dataset_preprocessing.py to preprocess facial and physiological data (although the physiological data is not required for unsupervised training). This will structure your data as follows:
	```bash
	PURE/
	  ├── Data/
	  │   ├── 01-01/
	  │   │   ├── video_mask.avi
	  │   │   └── phys.csv
	  │   ├── 01-02/
	  │   │   ├── video_mask.avi
	  │   │   └── phys.csv
	  │   └── ...
	  ├── Protocols/
	  │   └── Train.txt
 

4) **Train BeatFormer-SCL**

	Finally, run Train_SCL.py to start training BeatFormer-SCL.


If you find our paper or this code useful for your research, please cite our work.