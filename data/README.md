# Steam Dataset

This directory contains the Steam Video Game and Bundle Data from Professor Julian McAuley's research repository.

## Dataset Files

The dataset consists of the following files:

1. `reviews_v2.json.gz` (1.3GB): Contains review data including user IDs, item IDs, and playtime information.
2. `items_v2.json.gz` (2.7MB): Contains metadata about Steam games.
3. `bundles.json` (92KB): Contains information about game bundles.

## Downloading the Data

You can download the data by running the download script:

```bash
python scripts/download_data.py
```

Alternatively, you can manually download the files from the following URLs:

- Review Data: [https://snap.stanford.edu/data/steam/steam_reviews.json.gz](https://snap.stanford.edu/data/steam/steam_reviews.json.gz)
- Item Metadata: [https://snap.stanford.edu/data/steam/steam_games.json.gz](https://snap.stanford.edu/data/steam/steam_games.json.gz)
- Bundle Data: [https://snap.stanford.edu/data/steam/bundle_data.json.gz](https://snap.stanford.edu/data/steam/bundle_data.json.gz)

## Dataset Description

### Basic Statistics

- Reviews: 7,793,069
- Users: 2,567,538
- Items: 15,474
- Bundles: 615

### Data Format

Each file contains JSON objects, one per line. The format of each file is as follows:

#### Reviews (`reviews_v2.json.gz`)

```json
{
    "user_id": "U12345",
    "item_id": "I67890",
    "playtime_forever": 123,
    "playtime_2weeks": 45
}
```

#### Item Metadata (`items_v2.json.gz`)

```json
{
    "item_id": "I67890",
    "item_name": "Example Game",
    "genre": "Action, Adventure, RPG"
}
```

#### Bundles (`bundles.json`)

```json
{
    "bundle_id": "B12345",
    "bundle_name": "Example Bundle",
    "bundle_price": "$29.99",
    "bundle_final_price": "$19.99",
    "bundle_discount": "33%",
    "items": [
        {
            "item_id": "I67890",
            "item_name": "Example Game 1",
            "discounted_price": "$9.99",
            "genre": "Action, Adventure"
        },
        {
            "item_id": "I67891",
            "item_name": "Example Game 2",
            "discounted_price": "$14.99",
            "genre": "RPG, Strategy"
        }
    ]
}
```

## Reference

If you use this dataset in your research, please cite the following papers:

- **Self-attentive sequential recommendation** Wang-Cheng Kang, Julian McAuley *ICDM*, 2018
- **Item recommendation on monotonic behavior chains** Mengting Wan, Julian McAuley *RecSys*, 2018
- **Generating and personalizing bundle recommendations on Steam** Apurva Pathak, Kshitiz Gupta, Julian McAuley *SIGIR*, 2017