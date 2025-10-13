class PromptBuilder:
    """
    Utility class for building prompts for various LLM tasks.
    """

    @staticmethod
    def schema_description_prompt(db_schema: dict) -> str:
        return f"""
        {db_schema}

        These are the tables in the database.
        I want the schema description for each table and column.
        Return the output in JSON format like this:

        [
          {{
            "table": "table_name",
            "description": "short 1-line description of the table",
            "columns": [
              {{
                "name": "column_name",
                "type": "data_type",
                "description": "short 1-line description of the column"
              }}
            ]
          }}
        ]
        """.strip()
