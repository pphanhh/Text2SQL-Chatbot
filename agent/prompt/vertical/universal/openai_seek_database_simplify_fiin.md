<overall_description>

The database conatains financial statments of top Vietnamese firms, followed by the regulation of Vietnamese Accounting Standard (VAS). Based on VAS regulation, there are 3 type of financial statments: bank, corporation and securities firm, and there will be sight differences between them.

The database includes two reporting periods: quarterly and annually. Quarterly reports specify the relevant quarter (1, 2, 3, 4), whereas annual reports are indicated with the quarter set to 0.

</overall_description>

### PostgreSQL tables, with their properties, in OpenAI Template

<schema>

- **company_info**(stock_code, industry, exchange, stock_indices, is_bank, is_securities)
- **sub_and_shareholder**(stock_code, invest_on)
- **financial_statement**(stock_code, year, quarter, category_code, data, date_added)
- **industry_financial_statement**(industry, year, quarter, category_code, data_mean, data_sun, date_added)
- **financial_ratio**(ratio_code, stock_code, year, quarter, data, date_added)
- **industry_financial_ratio**(industry, ratio_code, year, quarter, data_mean, date_added)
- **financial_statement_explaination**(category_code, stock_code, year, quarter, data, date_added)

</schema>

### Note on schema description: 
- For `industry_` tables, column `data_mean` is average data of all firms in that industry, and `data_sum` is the sum of them.
- Table `financial_statement_explaination` contains information which is not covered in 3 main reports, usually about type of loans, debt, cash, investments and real-estate ownerships. 
- Each value in `category_code` includes a prefix indicating the report it pertains to: *BS* is Balance sheet, *IS* is for Income statement, *CF* is Cash flow and *TM* is for Explaination. For `category_code` in `financial_statement_explaination`, there are 4 additional prefix: *Crop*, *Bank*, *Sec* and *Share* for specific type of organization.
- With YoY ratio in `financial_ratio`, you should recalculate the ratio if the time window is not 1 year.

### Note on query:
- You will be provided a mapping table with caption for `category_code` and `ratio_code`, and you are only allow to use that reference to select suitable code.
- For any financial ratio, it must be selected from the database rather than being calculated manually.
- If two rows share a similar meaning, using a rounded code.
- Always include a `quarter` condition in your query. If not specified, assume using annual reports, with the query defaulting to `quarter` = 0.
- Always include a LIMIT clause in your query.