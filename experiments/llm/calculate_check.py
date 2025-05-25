import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
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

df = pd.read_pickle('E:/llm_check.pkl')


indexes_counted = [int(x.split("_")[0]) for x in os.listdir("check_llm")]
df = df[~df.index.isin(indexes_counted)]
PROGREESS_BAR = tqdm(total=df.shape[0])


error_file_lock = threading.Lock()
queue_lock = asyncio.Lock()
logger.remove()
logger.add(sys.stdout, level="ERROR")
WRITE = True
exception_counter = 0
MAX_EXCEPTIONS = 5


def do_safe_request(row: pd.Series, row_index, key_index) -> tuple[str, float]:
    # chunks_to_add: List[str] = row['texts']
    llm = GroqLLM(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct", api_key_num=key_index
    )
    while True:
        try:
            logger.info(f"В обработку поступил {row_index}")
            language = "ru" if row["dataset"] == "bench" else "en"
            # logger.info(language)
            start_time = time.time()
            # ---------------------------------------------------------------------------------------
            # РАСКОММЕНТИТЬ
            answer = llm.reveal_correctness(
                question=row["question"],
                ideal_answer=row["response"],
                llm_answer=row["answer"],
                lang=language,
            )
            # ---------------------------------------------------------------------------------------
            end_time = time.time()
            time.sleep(0.5)
        except APIStatusError as api_error:
            logger.warning("Переполнение контекста")
            if "Request too large" in str(api_error):
                # chunks_to_add.pop(-1)
                continue
            else:
                logger.error(api_error)
                # logger.error(traceback.format_exc())
                break
        except Exception as e:
            logger.error(
                f"Неизвестная ошибка на вопросе номер {row_index} с аккаунта index={key_index}"
            )
            logger.error(traceback.format_exc())
            # display(e)
            logger.info("Ждем минуту")
            time.sleep(60)
            exception_counter += 1
            if exception_counter >= MAX_EXCEPTIONS:
                with error_file_lock:
                    with open("ERRORS.txt", "a") as file:
                        file.write("У нас появился пятикратно переваренный кал\n")
                        file.write(f"на данных={str(row)}")
                        file.write(traceback.format_exc() + "\n\n\n\n")
                PROGREESS_BAR.update()
                return ({"correctness": -1, "completeness": -1}, 0.0)
            continue
        break
    logger.info(f"Успешно завершен {row_index}")
    PROGREESS_BAR.update()
    return answer, (end_time - start_time)


async def worker(queue: asyncio.Queue, key_index: int, executor: ThreadPoolExecutor):
    logger.info(f"worker номер {key_index} запущен")
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
            answer, generation_time = await loop.run_in_executor(
                executor, lambda: do_safe_request(row, row_index, key_index)
            )
            answer["generation_seconds"] = generation_time
            with open(
                f"check_llm/{row_index}_answer.json", "w", encoding="utf-8"
            ) as file:
                if WRITE:
                    json.dump(answer, file, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error in worker {key_index}: {str(e)}")
        finally:
            queue.task_done()


async def main(data, n_workers):
    logger.debug("Стартуем")
    queue = asyncio.Queue()
    for index, row in data.iterrows():
        await queue.put((index, row))
    logger.debug("Очередь засунута")
    # logger.debug(queue)
    logger.debug(f"{queue.qsize()=}")
    executor = ThreadPoolExecutor(max_workers=n_workers)
    tasks = [asyncio.create_task(worker(queue, i, executor)) for i in range(n_workers)]

    await queue.join()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    executor.shutdown()


if __name__ == "__main__":
    asyncio.run(main(df, n_workers=len(groq_api_keys)))
