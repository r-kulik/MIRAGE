import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import random
import re
import sys
import threading
import pandas as pd
import json
import time
import traceback
from typing import List
from groq import APIStatusError
from loguru import logger
from tqdm import tqdm
from LLMGroq import GroqLLM
from groq_keys import groq_api_keys

df = pd.read_pickle("E:\\llm_check.pkl")

indexes_counted = [int(x.split("_")[0]) for x in os.listdir("answers_llm")]
logger.info(f"{indexes_counted=}")
df = df[~df.index.isin(indexes_counted)]
PROGREESS_BAR = tqdm(total=df.shape[0])


error_file_lock = threading.Lock()
queue_lock = asyncio.Lock()
logger.remove()  # remove the old handler. Else, the old one will work (and continue printing DEBUG logs) along with the new handler added below'
logger.add(sys.stdout, level="ERROR")
exception_counter = 0
MAX_EXCEPTIONS = 5


class STOP_WORKER_RESPONSE:
    def __init__(self): ...


def text_cleaner(text):
    """
    Очищает текст от лишних пробелов, табуляции и переносов строк.

    Параметры:
        text (str): Исходный текст для очистки

    Возвращает:
        str: Очищенный текст
    """
    text = re.sub(r"[\t\n\r]", " ", text)
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def do_safe_request(row: pd.Series, row_index, key_index) -> tuple[str, float]:
    chunks_to_add: List[str] = row["texts"]
    chunks_to_add = list(map(text_cleaner, chunks_to_add))
    # logger.info(chunks_to_add)
    llm = GroqLLM(
        model=row['LLM_model'], api_key_num=key_index
    )
    while True:
        try:
            logger.info(f"В обработку поступил {row_index}")
            language = "ru" if row["dataset"] == "bench" else "en"
            start_time = time.time()
            answer = llm.do_request(
                query=row["question"], chunks=chunks_to_add, lang=language
            )
            print(answer)
            end_time = time.time()
            time.sleep(2)
        except APIStatusError as api_error:
            logger.warning("Переполнение контекста")
            if "Request too large" in str(api_error):
                chunks_to_add.pop(-1)
                continue
        except Exception as e:
            logger.error(
                f"Неизвестная ошибка на вопросе номер {row_index} с аккаунта index={key_index}"
            )
            logger.error(e)
            # display(e)
            logger.info("Ждем минуту")
            time.sleep(30)
            exception_counter += 1
            if exception_counter >= MAX_EXCEPTIONS:
                with error_file_lock:
                    with open("ERRORS.txt", "a") as file:
                        file.write("У нас появился пятикратно переваренный кал\n")
                        file.write(f"на данных={str(row)}")
                        file.write(traceback.format_exc() + "\n\n\n\n")
                PROGREESS_BAR.update()
                return ("NONE", 0.0)
            continue
        break
    logger.info(f"Успешно завершен {row_index}")

    try:
        return answer, (end_time - start_time)
    except UnboundLocalError:
        return (STOP_WORKER_RESPONSE(), 0)


async def worker(queue: asyncio.Queue, key_index: int, executor: ThreadPoolExecutor):
    time.sleep(random.random() * 5)
    logger.error(f"worker номер {key_index} запущен")
    while True:

        logger.info(f"ИЩУ РАБОТУ for {key_index}")
        async with queue_lock:  # Блокируем доступ к очереди
            if queue.empty():
                logger.info(f"Очередь пуста для worker {key_index}")
                break
            row_index, row = await queue.get()

        logger.info(f"ОПЯТЬ РАБОТА for {key_index}: {row_index}")
        loop = asyncio.get_event_loop()
        try:
            logger.info("Запускаем executor")
            # logger.warning(row)
            answer, generation_time = await loop.run_in_executor(
                executor, lambda: do_safe_request(row, row_index, key_index)
            )
            if isinstance(answer, STOP_WORKER_RESPONSE):
                async with queue_lock:
                    await queue.put((row_index, row))
                logger.error(f"STOPPING WORKER #{key_index}")
                return None
            # Save the result
            result = {
                "question": row["question_text"],
                "index_file": row["file"],
                "question_id": row["question"],
                "answer": answer,
                "generation_seconds": generation_time,
                "k1": row["k1"],
                "c": row["c"],
            }
            PROGREESS_BAR.update()
            # Save periodically
            with open(
                f"answers_llm/{row_index}_answer.json", "w", encoding="utf-8"
            ) as file:
                json.dump(result, file, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error in worker {key_index}: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            queue.task_done()


async def main(data, n_workers):
    counter = 0
    MAX_ELEMENTS = float("inf")
    logger.debug("Стартуем")
    queue = asyncio.Queue()
    for index, row in data.iterrows():
        await queue.put((index, row))
        counter += 1
        if counter > MAX_ELEMENTS:
            break
    logger.debug("Очередь засунута")
    # logger.debug(queue)
    logger.debug(f"{queue.qsize()=}")
    executor = ThreadPoolExecutor(max_workers=n_workers)
    tasks = [asyncio.create_task(worker(queue, i, executor)) for i in range(n_workers)]

    await queue.join()

    # Cancel all worker tasks
    for task in tasks:
        task.cancel()
    # Wait until all worker tasks are cancelled
    await asyncio.gather(*tasks, return_exceptions=True)
    executor.shutdown()


LAST_ELEM = True
if __name__ == "__main__":
    asyncio.run(main(df, n_workers=len(groq_api_keys)))
