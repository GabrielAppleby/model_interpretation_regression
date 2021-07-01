import os
from pathlib import Path
from typing import Dict, List

import psycopg2
from dotenv import load_dotenv
from psycopg2._psycopg import connection, cursor

from data.data_config import DATA_FOLDER, FULL_CSV_FILE_NAME, NCT_ID, NUM_SAE

load_dotenv()

DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
HOST = os.environ.get('HOST')
PASSWORD = os.environ.get('PASSWORD')

SCHEMA = 'ctgov'
MAIN_TABLE = 'studies'
SCHEMA_MAIN_TABLE = '{}.{}'.format(SCHEMA, MAIN_TABLE)

TABLES_AND_FIELDS: Dict[str, List[str]] = {MAIN_TABLE:
                                               [NCT_ID,
                                                'phase',
                                                'enrollment',
                                                'number_of_arms',
                                                'has_expanded_access'],
                                           'calculated_values':
                                               [NUM_SAE,
                                                'number_of_facilities',
                                                'actual_duration',
                                                'months_to_report_results',
                                                'minimum_age_num',
                                                'minimum_age_unit',
                                                'number_of_primary_outcomes_to_measure',
                                                'number_of_secondary_outcomes_to_measure'],
                                           'browse_interventions': ['mesh_term'],
                                           'browse_conditions': ['mesh_term']}

RESTRAINTS = ['overall_status = \'Completed\'',
              'study_type = \'Interventional\'',
              '{} IS NOT NULL'.format(NUM_SAE)]

QUERY_STATEMENT = 'SELECT {fields} FROM {table} {table_joins} WHERE {restraints}'

WRITE_OUTPUT_STATEMENT = 'COPY ({}) TO STDOUT WITH CSV HEADER'


def get_qualified_fields(tables_and_fields: Dict[str, List[str]]) -> List[str]:
    return ['{}.{}'.format(table, field) for table in tables_and_fields for field in
            tables_and_fields[table]]


def get_table_joins(tables_and_fields: Dict[str, List[str]], schema: str, main_table: str,
                    nct_id: str) -> List[str]:
    return ['INNER JOIN {schema}.{nt} ON {ot}.{id} = {nt}.{id}'.format(schema=schema, nt=table,
                                                                       ot=main_table, id=nct_id)
            for table in list(tables_and_fields.keys())[1:]]


def get_fields_str(qualified_fields: List[str]) -> str:
    return ", ".join(qualified_fields)


def get_table_joins_str(table_joins: List[str]) -> str:
    return " ".join(table_joins)


def get_restraints_str(restraints: List[str]) -> str:
    return " AND ".join(restraints)


def get_db_connection(dbname: str, user: str, host: str, password: str) -> connection:
    return psycopg2.connect(dbname=dbname,
                            user=user,
                            host=host,
                            password=password)


def main():
    DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    conn: connection = get_db_connection(DB_NAME, DB_USER, HOST, PASSWORD)
    cur: cursor = conn.cursor()

    qualified_fields = get_qualified_fields(TABLES_AND_FIELDS)
    table_joins = get_table_joins(TABLES_AND_FIELDS, SCHEMA, MAIN_TABLE, NCT_ID)
    fields_str = get_fields_str(qualified_fields)
    table_joins_str = get_table_joins_str(table_joins)
    restraints_str = get_restraints_str(RESTRAINTS)
    query = QUERY_STATEMENT.format(fields=fields_str, table=SCHEMA_MAIN_TABLE,
                                   table_joins=table_joins_str, restraints=restraints_str)

    output_query = WRITE_OUTPUT_STATEMENT.format(query)

    with open(Path(DATA_FOLDER, FULL_CSV_FILE_NAME), 'w') as f:
        cur.copy_expert(output_query, f)

    cur.close()


if __name__ == '__main__':
    main()
