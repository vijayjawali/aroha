The Following flow represents the folder structure of the project 

				aroha
				|       
				+---bart
				|   |   bart.csv.zip
				|   |   bart.ipynb
				|   |   bart_rouge.csv
				|   |   
				|   \---markdown_files
				|           ...
				|           
				+---dash_app
				|       config.txt
				|       dash_app.py
				|       run_dash_app.ipynb
				|       
				+---eda
				|   |   article.parquet
				|   |   articles_string.zip
				|   |   bart.csv.zip
				|   |   eda.ipynb
				|   |   
				|   \---named_entities
				|           batch_1.parquet
				|           ...
				|           batch_312.parquet
				|           
				+---llama2
				|   |   llama2.csv
				|   |   llama2.csv.zip
				|   |   llama2.ipynb
				|   |   llama2_rouge.csv
				|   |   
				|   \---llama_outputs
				|           batch_1.csv.zip
				|           ...
				|           batch_115.csv.zip
				|           
				+---llama2_fine_tuned
				|   |   llama2_fine_tuned.csv.zip
				|   |   llama2_fine_tuned_rouge.csv
				|   |   llama2_fine_tuned_test.ipynb
				|   |   llama2_fine_tuned_trainer.ipynb
				|   |   
				|   +---llama2_fine_tuned_outputs
				|   |       batch_1.csv.zip
				|   |       ...
				|   |       batch_115.csv.zip
				|   |       
				|   \---llama2_query_tuning
				|       +---logs
				|       |       ...
				|       |       
				|       +---model
				|       |       adapter_config.json
				|       |       adapter_model.bin
				|       |       README.md
				|       |       special_tokens_map.json
				|       |       tokenizer.json
				|       |       tokenizer.model
				|       |       tokenizer_config.json
				|       |       training_args.bin
				|       |       
				|       \---results
				|           +---checkpoint-10
				|           |       adapter_config.json
				|           |       adapter_model.bin
				|           |       optimizer.pt
				|           |       README.md
				|           |       rng_state.pth
				|           |       scheduler.pt
				|           |       special_tokens_map.json
				|           |       tokenizer.json
				|           |       tokenizer.model
				|           |       tokenizer_config.json
				|           |       trainer_state.json
				|           |       training_args.bin
				|           |       
				|           |       
				|           \---checkpoint-250
				|                   ...
				|                   
				+---lsa
				|       lsa.csv.zip
				|       lsa.ipynb
				|       lsa_rouge.csv
				|       
				+---seq2seq_pg
				|   |   article_tokenizer.pickle
				|   |   article_vocabulary.json
				|   |   decoder_model.h5
				|   |   decoder_weights.h5
				|   |   encoder_model.h5
				|   |   encoder_weights.h5
				|   |   params.txt
				|   |   seq2seq_pg.csv.zip
				|   |   seq2seq_pg.h5
				|   |   seq2seq_pg.ipynb
				|   |   seq2seq_pg_rouge.csv
				|   |   seq2seq_test.ipynb
				|   |   summary_tokenizer.pickle
				|   |   summary_vocabulary.json
				|   |   
				|   +---markdown_files
				|   |       ...
				|   |       
				|   \---seq2seq_outputs
				|           batch_1.csv.zip
				|           ...
				|           batch_115.csv.zip
				|           
				+---t5
				|   |   t5.csv.zip
				|   |   t5.ipynb
				|   |   t5_rouge.csv
				|   |   
				|   +---markdown_files
				|   |       ...
				|   |       
				|   \---t5_outputs
				|           batch_1.csv.zip
				|           ...
				|           batch_115.csv.zip
				|           
				+---t5_fine_tuned
				|   |   t5_transfer_learning.csv.zip
				|   |   t5_transfer_learning_predictions.ipynb
				|   |   t5_transfer_learning_rouge.csv
				|   |   t5_transfer_learning_trainer.ipynb
				|   |   
				|   \---t5_transfer_learning
				|       +---logs
				|       |   ...
				|       +---model
				|       |       config.json
				|       |       generation_config.json
				|       |       pytorch_model.bin
				|       |       training_args.bin
				|       |       
				|       \---results
				|           +---checkpoint-143558
				|           |       config.json
				|           |       generation_config.json
				|           |       optimizer.pt
				|           |       pytorch_model.bin
				|           |       rng_state.pth
				|           |       scheduler.pt
				|           |       trainer_state.json
				|           |       training_args.bin
				|           |       
				|           +---checkpoint-215337
				|           |       ...
				|           |       
				|           \---checkpoint-71779
				|                   ...
				|                   
				+---text_rank
				|   |   text_rank.csv.zip
				|   |   text_rank.ipynb
				|   |   text_rank_rouge.csv
				|   |   
				|   \---markdown_files
				|           ...
				|           
				+---tf_idf
				|   |   tf_idf.csv.zip
				|   |   tf_idf.ipynb
				|   |   tf_idf_rouge.csv
				|   |   
				|   \---markdown_files
				|           ...
				|           
				\---utils
						meteor_scores.csv
						metrics.ipynb

1. Fist step is to install the following python libraries from the pip command

	python -m pip install -U pip
	pip install -r requirements.txt
	
	a. To run individual summarisation model, navigate to the specific folder and install packages from the specific requirements.txt file
	
	b. To run dashboard application, navigate to dash_app folder and execute requirements.txt
	
2. Download CNN/Dailymail dataset from HuggingFace using either the curl command or python library
	
	a. Curl Command : curl -X GET \
     "https://datasets-server.huggingface.co/splits?dataset=cnn_dailymail"
	b. Python Library :
		from datasets import load_dataset
		dataset = load_dataset('cnn_dailymail', '3.0.0')

3. Configure config.txt file
	
	a. Navigate to config.txt file in dash_app folder
	b. Replace the configuration parameters with the desired paths
	
	named_entities,/content/drive/MyDrive/aroha/eda/named_entities/
	bart_tokenizer_cache_dir_path,/content/drive/MyDrive/aroha/bart_cache
	bart_model_cache_dir_path,/content/drive/MyDrive/aroha/bart_cache
	t5_tokenizer_cache_dir_path,/content/drive/MyDrive/aroha/t5_cache  
	t5_model_cache_dir_path,/content/drive/MyDrive/aroha/t5_cache
	t5_fine_tuned_model_path,/content/drive/MyDrive/aroha/t5_transfer_learning/model
	llama2_access_token_path,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
	llama2_tokenizer_cache_dir_path,/content/drive/MyDrive/aroha/llama2_cache
	llama2_model_cache_dir_path,/content/drive/MyDrive/aroha/llama2_cache
	peft_model_path,/content/drive/MyDrive/aroha/llama2_fine_tuned/model
	seq2seq_encoder_model_path,/content/drive/MyDrive/aroha/seq2seq_pg/encoder_model.h5
	seq2seq_decoder_model_path,/content/drive/MyDrive/aroha/seq2seq_pg/decoder_model.h5
	summary_tokenizer_path,/content/drive/MyDrive/aroha/seq2seq_pg/summary_tokenizer.pickle
	article_tokenizer_path,/content/drive/MyDrive/aroha/seq2seq_pg/article_tokenizer.pickle
	summary_vocabulary_path,/content/drive/MyDrive/aroha/seq2seq_pg/summary_vocabulary.json
	
	c. Contents for "named_entities" is available in eda folder
	d. Contents for "t5_fine_tuned_model_path" ia available at "t5_fine_tuned\t5_transfer_learning\model"
	e. "llama2_fine_tuned" model has been published to HuggingFace and can be downloaded directly

	config = PeftConfig.from_pretrained("vijayjawali/llama2_query_tuned")
	model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
	model = PeftModel.from_pretrained(model, "vijayjawali/llama2_query_tuned")
	
	f. Contents for summary_tokenizer_path, article_tokenizer_path and summary_vocabulary_path are avilable in seq2seq_pg folder
	
4. Get Llama-2 Aceess from Meta and HuggingFace
	
	a. Open Request Form : https://ai.meta.com/resources/models-and-libraries/llama-downloads/
	b. Enter details and wait to get access, it should take 2-3 hours
	c. Request the same from HuggingFace : https://huggingface.co/meta-llama
	d. After having both the access, navigate to settings page in HuggingFace and create a read token: https://huggingface.co/settings/tokens
	e. Replace the config parameter llama2_access_token_path with the token received
	
	
5. After the configuration file is setup, navigate to dash_app folder, it contains two files, dash_app.py and run_dash_app.ipynb
	
	a. copy dash_app.py and place it in the execution environment
	b. open run_dash_app.ipynb and navigate to last line containing python execution code
	
	! python /content/drive/MyDrive/aroha/dash_app/dash_app.py
	
	c. replace the python execution file location with dash_app.py path
	d. run_dash_app executes the dash_app.py file to run dashboard application

6. Get Ngrok authtoken from https://dashboard.ngrok.com/auth

	a. Replace the existing token with the one generated from ngrok
	
	NGROK_AUTH_TOKEN = "xxxxxxxxxxxxxxxxxxxxxx" 
	pyngrok.set_auth_token(NGROK_AUTH_TOKEN)

7. Execute run_dash_app.ipynb file
	
	a. It is recommended to use NVIDIA A100 in colab for smooth execution.
	b. Alternative T4 GPU can be used to execute.
	c. Copy public url logged by ngrok
	
	logger.info(f"Dash app:{ngrok_tunnel.public_url}")
	
	d. Open the copied link in browser to access dashboard application
	