define n


endef

all:
	$(error "Plese choose between 'make content' and 'make collaborative'")

content: dataset/anime_merged.csv
	@pip install -r requirements.txt 1>/dev/null || (echo "pip install failed" && exit 1)
	python3 src/content_based_recommender.py

collaborative: dataset/anime.csv dataset/rating_complete.csv
	@pip install -r requirements.txt 1>/dev/null || (echo "pip install failed" && exit 1)
	python3 src/collaborative_filtering_recommender.py

dataset/anime_merged.csv: dataset/anime.csv dataset/anime_with_synopsis_keywords.csv
	python3 src/merge_datasets.py

dataset/anime_with_synopsis_keywords.csv:
	python3 src/synopsis_keywords.py

uninstall_nltk:
	@echo "Uninstalling nltk from home directory"
	rm -rf ~/nltk_data

dataset/%: | archive.zip
	mkdir ./dataset
	unzip ./archive.zip -d ./dataset/

archive.zip:
	$(error $narchive.zip not found \
	$nPlease download the dataset from https://www.kaggle.com/hernan4444/anime-recommendation-database-2020/version/7 \
	$nand place it in the root directory of this project )
	
zip:
	@echo "Zipping up the project"
	rm -f project.zip
	zip -r project.zip src .gitignore LICENSE README.md Makefile requirements.txt

zip-all:
	@echo "Zipping up the project with heavy files"
	zip -r project.zip .