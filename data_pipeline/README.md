# data_pipeline

A learning project to build a simple data pipeline. It scrapes book listings from
books.toscrape.com (a practice site made for this kind of work) across four
categories, cleans the data a bit, and stores it in two places - a csv file and a
SQLite database.

The pipeline code lives in `cv.ipynb`. Right now this is mostly a working draft,
not a finished product.

## What it does

1. Pulls category pages (Travel, Mystery, Historical Fiction, Sequential Art) from
   books.toscrape.com
2. Extracts book name, rating, price and stock info using requests + BeautifulSoup
3. Converts prices from GBP to INR and ratings from words (like "Four") to numbers
   because that's easier to work with later
4. Writes the results to `cv.csv`
5. Loads the same data into a SQLite database (`tutorial.db`) with `categories`
   and `books` tables, linked by a foreign key

## Files

- `cv.ipynb` - the notebook that does all the scraping and storing
- `cv.csv` - scraped book data (72 rows)
- `tutorial.db` - SQLite database with the scraped data

## Running it

The notebook expects a few packages. Install them with:

```bash
pip install requests beautifulsoup4 pandas
```

Then open the notebook:

```bash
jupyter notebook cv.ipynb
```

Run the cells in order. The scraping calls the live website, so it will only work
while books.toscrape.com is up.

## Notes

- Prices are hardcoded to convert at roughly 105.50 INR per GBP, which is nowhere
  near a real exchange rate - it's just there to practice currency conversion.
- The site only shows one page per category here, so it does not go through every
  page of the catalogue.
- No license yet, so treat the code as all-rights-reserved.