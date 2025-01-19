

import pandas as pd  # type: ignore
from typing import List
from Judgment_module.LLM_judge import JudgmentLLM
import logging # type: ignore
import asyncio
import os

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class DataHandler:
    """
    A class to handle reading and processing data from files,
    and operating on DataFrames.
    """

    def __init__(self,judgment):
        """
        Initialize the DataHandler instance with default settings.
        """
        self.columns: List[str] = []
        if not isinstance(judgment, JudgmentLLM):
            raise TypeError("you must pass an instance of JudgmentLLM")
        else:
            self.judgment = judgment

    def read_csv(self, file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and returns its contents as a DataFrame.

        Parameters:
            file_path (str): The path to the CSV file to be read.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the contents of the CSV file.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            pd.errors.EmptyDataError: If the file is empty or contains no data.
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            logger.error(f"CSV file not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {file_path}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading CSV: {e}")
            raise

    def read_excel(self, file_path: str) -> pd.DataFrame:
        """
        Reads an Excel file and returns its contents as a DataFrame.

        Parameters:
            file_path (str): The path to the Excel file to be read.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the contents of the Excel file.

        Raises:
            FileNotFoundError: If the Excel file does not exist.
            ValueError: If no sheets are found or the file is empty.
        """
        try:
            df = pd.read_excel(file_path)
            return df
        except FileNotFoundError:
            logger.error(f"Excel file not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Excel file is empty: {file_path}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading Excel: {e}")
            raise


    def specify_columns(self, columns: list[str]) -> None:
            """
            Specify the columns for ground truth and predicted values.

            Parameters:
                columns (list[str]): A list containing the names of the columns for ground truth and predicted values.
                                    The first element is the ground truth column, and the second is the predicted column.
            """
            if self.judgment.user_prompt == True:
                if self.judgment.option == "discrepancy_judgment":
                    if not columns or len(columns) != 3:
                        raise ValueError("The 'columns' parameter must be a list containing exactly three column names [ user_query_that_lead_to_statements, statement_1, statement_2].")
                    self.columns.extend(columns)
                else:
                    if not columns or len(columns) != 2:
                        raise ValueError("The 'columns' parameter must be a list containing exactly two column names  [ statement, user_query_that_lead_to_the_statement].")
                    self.columns.extend(columns)
            if self.judgment.user_prompt == False:

                if self.judgment.option == "discrepancy_judgment":
                    if not columns or len(columns) != 2:
                        raise ValueError("The 'columns' parameter must be a list containing exactly two column names [statement_1, statement_2].")
                    self.columns.extend(columns)
                else:
                    if not columns or len(columns) != 1:
                        raise ValueError("The 'columns' parameter must be a list containing exactly one column name.[statement]")
                    self.columns.extend(columns)

            
            

    def decompose_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a new DataFrame with the specified columns in `self.columns`.

        Parameters:
            df (pd.DataFrame): The DataFrame to be decomposed.

        Returns:
            pd.DataFrame: A new DataFrame containing only the specified columns.

        Raises:
            ValueError: If no columns have been specified or if specified
                        columns are not present in the DataFrame.
        """
        if not self.columns:
            raise ValueError("Columns have not been specified. Please call `specify_columns()` first.")
        
        missing_columns = [col for col in self.columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following specified columns are not in the DataFrame: {missing_columns}")
        
        return df[self.columns]


    def proccess_df(self, df: pd.DataFrame) -> None:
        results = []

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            row_values_list = []
            # Convert the row to a list and append it to the row_values_list
            row_values_list.extend(row.tolist())
            res = self.judgment.generate_judgment(row_values_list)
            results.append(res)
        return results




    async def process_df_async(self, df: pd.DataFrame) -> None:
        """
        Asynchronously process the DataFrame in parallel.
        """
        tasks = self.gather_tasks(df)
        # Run all the tasks concurrently
        results = await asyncio.gather(*tasks)
        return results
    
    def gather_tasks(self,df):
        tasks = []

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            row_values_list = []
            row_values_list.extend(row.tolist())

            
            # Create an asyncio task for processing the row asynchronously
            task = asyncio.create_task(self.judgment.generate_judgment_async(row_values_list))
            tasks.append(task)
        return tasks
    async def run_judgment_pipline_async(self, input_path,string_list,save_path):
        df = self.read_file(input_path)
        self.specify_columns(string_list)
        df_new = self.decompose_df(df)
        res =  await self.process_df_async(df_new)
        df_new = self.create_judgment_df(res,df_new)
        self.save_df(df_new,save_path)
        return res,df_new
    
    def run_judgment_pipeline(self,input_path,string_list,save_path):
        df = self.read_file(input_path)
        self.specify_columns( string_list)
        df_new = self.decompose_df(df)
        res =  self.process_df(df_new)
        df_new = self.create_judgment_df(res,df_new)
        self.save_df(df_new,save_path)
        
        return res,df_new

    
    def create_judgment_df(self,results,df):
        new_df = df.copy()
        new_df['judgment_reasoning'] = [item.reasoning for item in results]
        new_df['judgment_total_score'] = [item.total_score.value for item in results]
        return new_df


    def save_df(self,df,path):
        df.to_csv(path, index=False)


    
    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        Reads a file (CSV or Excel) and returns its contents as a DataFrame.

        Parameters:
            file_path (str): The path to the file to be read.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the contents of the file.

        Raises:
            ValueError: If the file extension is not supported.
        """
        # Extract the file extension
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        try:
            # Check the file extension and read accordingly
            if file_extension == ".csv":
                return self.read_csv(file_path)
            elif file_extension in [".xls", ".xlsx"]:
                return self.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Only '.csv' and '.xls/.xlsx' are supported.")
        except Exception as e:
            logger.error(f"Error while reading the file: {e}")
            raise