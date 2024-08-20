import pandas as pd
from image_search_engine.src import config
import logging

logger = logging.getLogger(__name__)

def processing_dataset(dataset_file: str) -> pd.DataFrame:
    df = pd.read_csv(config.RAW_PATH / dataset_file, delimiter="|")
    df.columns = ['image', 'caption_number', 'caption']
    df['caption'] = df['caption'].str.lstrip()
    df['caption_number'] = df['caption_number'].str.lstrip()
    df.loc[19999, 'caption_number'] = "4"
    df.loc[19999, 'caption'] = "A dog runs across the grass ."
    ids = [id_ for id_ in range(len(df) // 5) for i in range(5)]
    df['id'] = ids
    return df

def save_processed_data(data: pd.DataFrame) -> None:
    logger.info("Saving data...")
    data.to_csv(config.PROCESSED_PATH / "captions.csv", index=False)
    logger.info("Data saved as csv.")
    
if __name__ == "__main__":
    dataset = processing_dataset("results.csv")
    save_processed_data(dataset)