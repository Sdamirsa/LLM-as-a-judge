

import pandas as pd  # type: ignore
from typing import List,Union, Optional
from Judgment_module.LLM_judge import JudgmentLLM
import logging # type: ignore
import asyncio
import os

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class DataHandler:
    """
            A class to handle reading and processing data from various input sources,
            including files and DataFrames.

            Input Types Supported:
            1. pandas DataFrame: Pass your existing DataFrame directly
            2. CSV files: Provide path to a .csv file
            3. Excel files: Provide path to a .xls or .xlsx file

            The class can either:
            - Automatically detect the input type from file extensions
            - Use explicitly specified input type via the input_type parameter
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
    
    def read_input(self, input_source: Union[str, pd.DataFrame], input_type: Optional[str] = None) -> pd.DataFrame:
        """
        Reads data from various input sources and returns a DataFrame.

        Parameters:
            input_source (Union[str, pd.DataFrame]): The input source. Can be either:
                - A pandas DataFrame object
                - A string path to a CSV file
                - A string path to an Excel file (.xls or .xlsx)
            
            input_type (Optional[str]): Type of input. Allowed values:
                - 'csv': For CSV files
                - 'excel': For Excel files
                - 'dataframe': For pandas DataFrame objects
                - None: Will attempt to auto-detect from file extension

        Returns:
            pd.DataFrame: A pandas DataFrame containing the data

        Raises:
            ValueError: If input type is not supported or cannot be determined
            FileNotFoundError: If the specified file does not exist
            pd.errors.EmptyDataError: If the input file is empty

        Examples:
            # Using with a DataFrame
            df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
            handler.read_input(df)

            # Using with a CSV file (auto-detect)
            handler.read_input('path/to/file.csv')

            # Using with an Excel file (explicit type)
            handler.read_input('path/to/file.xlsx', input_type='excel')
        """
        if isinstance(input_source, pd.DataFrame):
            return input_source

        if isinstance(input_source, str):
            if input_type is None:
                # Try to infer input type from file extension
                _, file_extension = os.path.splitext(input_source)
                file_extension = file_extension.lower()
                
                if file_extension == '.csv':
                    input_type = 'csv'
                elif file_extension in ['.xls', '.xlsx']:
                    input_type = 'excel'
                else:
                    raise ValueError(
                        f"Could not determine input type from extension: {file_extension}. "
                        "Please explicitly specify input_type as 'csv' or 'excel'"
                    )

            try:
                if input_type == 'csv':
                    return self.read_csv(input_source)
                elif input_type == 'excel':
                    return self.read_excel(input_source)
                else:
                    raise ValueError(
                        f"Unsupported input type: {input_type}. "
                        "Supported types are: 'csv', 'excel', or pandas DataFrame"
                    )
            except Exception as e:
                logger.error(f"Error reading input: {e}")
                raise

        raise ValueError(
            "Input source must be either a pandas DataFrame or a file path string. "
            "For files, supported formats are CSV (.csv) and Excel (.xls, .xlsx)"
        )

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
                if self.judgment.option == "ground_truth_judgment":
                    if not columns or len(columns) != 2:
                        raise ValueError("The 'columns' parameter must be a list containing exactly two column names [statement_1, ground_truth].")
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
    async def run_judgment_pipline_async(self, 
                                       input_source: Union[str, pd.DataFrame],
                                       string_list: List[str],
                                       save_path: str,
                                       input_type: Optional[str] = None) -> tuple:
        """
        Run the async judgment pipeline with flexible input handling.

        Parameters:
            input_source (Union[str, pd.DataFrame]): The input source. Can be:
                - A pandas DataFrame object
                - A string path to a CSV file
                - A string path to an Excel file
            string_list (List[str]): List of column names to process
            save_path (str): Path where the output CSV will be saved
            input_type (Optional[str]): Type of input ('csv', 'excel', 'dataframe').
                                      If None, will try to infer from file extension.

        Returns:
            tuple: (results, processed_dataframe)

        Examples:
            # Using with a CSV file
            results, df = await handler.run_judgment_pipline_async(
                'data.csv',
                ['col1', 'col2'],
                'output.csv'
            )

            # Using with a DataFrame
            results, df = await handler.run_judgment_pipline_async(
                my_dataframe,
                ['col1', 'col2'],
                'output.csv'
            )
        """
        df = self.read_input(input_source, input_type)
        self.specify_columns(string_list)
        df_new = self.decompose_df(df)
        res = await self.process_df_async(df_new)
        df_new = self.create_judgment_df(res, df_new)
        self.save_df(df_new, save_path)
        return res, df_new

    def run_judgment_pipeline(self,
                            input_source: Union[str, pd.DataFrame],
                            string_list: List[str],
                            save_path: str,
                            input_type: Optional[str] = None) -> tuple:
        """
        Run the synchronous judgment pipeline with flexible input handling.

        Parameters:
            input_source (Union[str, pd.DataFrame]): The input source. Can be:
                - A pandas DataFrame object
                - A string path to a CSV file
                - A string path to an Excel file
            string_list (List[str]): List of column names to process
            save_path (str): Path where the output CSV will be saved
            input_type (Optional[str]): Type of input ('csv', 'excel', 'dataframe').
                                      If None, will try to infer from file extension.

        Returns:
            tuple: (results, processed_dataframe)

        Examples:
            # Using with an Excel file
            results, df = handler.run_judgment_pipeline(
                'data.xlsx',
                ['col1', 'col2'],
                'output.csv'
            )

            # Using with a DataFrame, explicit type
            results, df = handler.run_judgment_pipeline(
                my_dataframe,
                ['col1', 'col2'],
                'output.csv',
                input_type='dataframe'
            )
        """
        df = self.read_input(input_source, input_type)
        self.specify_columns(string_list)
        df_new = self.decompose_df(df)
        res = self.process_df(df_new)
        df_new = self.create_judgment_df(res, df_new)
        self.save_df(df_new, save_path)
        return res, df_new

    
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