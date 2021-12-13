import pandas as pd
from text_normalization import normalizeTextToKeywords
from nltk_resources import download_all_nltk_resources_if_needed
download_all_nltk_resources_if_needed() # TODO: download only needed resources

import cols

ANIME_SYNOPSIS_FILE = 'dataset/anime_with_synopsis.csv'
ANIME_SYNOPSIS_KEYWORDS_FILE = 'dataset/anime_with_synopsis_keywords.csv'

def generateSynopsisKeywords():
    print("[GenerateSynopsisKeywords] Generating synopsis keywords...")

    ORIG_SYNOPSIS = 'sypnopsis'

    print("[GenerateSynopsisKeywords] Reading synopsis...")
    cols_to_use = [cols.MAL_ID, ORIG_SYNOPSIS]
    anime_synopsis_df = pd.read_csv(ANIME_SYNOPSIS_FILE, usecols = cols_to_use)

    print("[GenerateSynopsisKeywords] Dropping missing values...")
    anime_synopsis_df.dropna(inplace = True) # Drop rows with missing values

    # null_lines = anime_synopsis_df[(anime_synopsis_df[ORIG_SYNOPSIS].isnull() == True)]
    # print('Number of non-null lines: ', len(null_lines))
    # Output: Number of non-null lines:  0

    # no_synopsis_lines = anime_synopsis_df[anime_synopsis_df[ORIG_SYNOPSIS].str.contains('No synopsis')]
    # print('Number of lines with no synopsis: ', len(no_synopsis_lines))
    # Output: Number of lines with no synopsis:  745

    print("[GenerateSynopsisKeywords] Removing 'No synopsis'...")
    no_synopsis_filter = ~anime_synopsis_df[ORIG_SYNOPSIS].str.contains('No synopsis')
    anime_synopsis_df = anime_synopsis_df[no_synopsis_filter]

    print("[GenerateSynopsisKeywords] Normalizing synopsis...")
    anime_synopsis_df['Synopsis_Keywords'] = anime_synopsis_df[ORIG_SYNOPSIS].apply(normalizeTextToKeywords)

    print(f"[GenerateSynopsisKeywords] Renamimg column '{ORIG_SYNOPSIS}' to '{cols.SYNOPSIS}'...")
    anime_synopsis_df.rename(columns = {ORIG_SYNOPSIS: cols.SYNOPSIS}, inplace = True)

    # Overwrite the original file with the new column (synopsis_keywords)
    anime_synopsis_df.to_csv(ANIME_SYNOPSIS_KEYWORDS_FILE, index = False)
    print("[GenerateSynopsisKeywords] Done.")


if __name__ == '__main__':
    generateSynopsisKeywords()