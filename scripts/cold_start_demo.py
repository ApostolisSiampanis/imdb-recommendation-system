from src.inference import recommend_for_new_user
import json

def main():
    watchlist_items = [123456, 345678, 456789]
    watchlist_ratings = {123456: 9, 456789: 10}

    recs = recommend_for_new_user(
        watchlist_items=watchlist_items,
        watchlist_ratings=watchlist_ratings,
        top_k=8,
        min_rating_pred=1.0,
        default_watchlist_rating=8,
        return_internal_ids=True
    )
    print(json.dumps(recs, indent=2))

if __name__ == "__main__":
    main()