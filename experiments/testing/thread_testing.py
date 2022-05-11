import logging
import threading
import time

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

def population_thread_function(population,agent, round):
    logging.info(f"population: {population}, agent: {agent}, round: {round}")
    time.sleep(0.1)
    logging.info(f"Finish population: {population}, agent: {agent}, round: {round}")


def agent_thread_function(agent, delay, round):
    logging.info(f"Agent: {agent}, round: {round}")
    time.sleep(delay)
    # In each thread create 2 threads (rounds)
    threads = []
    for n in range(5):
        threads.append(threading.Thread(target=population_thread_function, args=(n,agent,round, )))
        threads[-1].start()
    
    for e, thread in enumerate(threads):
        logging.info(f"Joing agent: {agent}, thread: {e}, round: {round}")
        thread.join()
    logging.info(f"Finish Agent: {agent}, round: {round}")


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    # logging.info("Main    : before creating thread")
    # x = threading.Thread(target=thread_function, args=(1,))
    # logging.info("Main    : before running thread")
    # x.start()
    # logging.info("Main    : wait for the thread to finish")
    # # x.join()
    # logging.info("Main    : all done")

    # For loop for the rounds
    delay = [0.1,0.1]   # pred will take 5 seconds and prey will take 1 second
    for i in range(50):
        logging.info(f"Start round: {i}")
        threads = []
        for e,agent in enumerate(["pred", "prey"]):
            # Create thread for each agent (2 thread)
            threads.append(threading.Thread(target=agent_thread_function, args=(agent,delay[e],i,)))
            threads[-1].start()
        # Wait for both the threads to join and finish
        for e, thread in enumerate(threads):
            logging.info(f"Joing round: {i}, thread: {e}")
            thread.join()
        print("--------------------------------------------------")



# 16:51:37: Start round #:0
# 16:51:37: Agent: pred, round: 0
# 16:51:37: Agent: prey, round: 0
# 16:51:37: Joing round: 0, thread: 0
# 16:51:38: population: 0, agent: prey, round: 0
# 16:51:38: population: 1, agent: prey, round: 0
# 16:51:38: Joing agent: prey, thread: 0, round: 0
# 16:51:40: Finish population: 0, agent: prey, round: 0
# 16:51:40: Finish population: 1, agent: prey, round: 0
# 16:51:40: Joing agent: prey, thread: 1, round: 0
# 16:51:40: Finish Agent: prey, round: 0
# 16:51:42: population: 0, agent: pred, round: 0
# 16:51:42: population: 1, agent: pred, round: 0
# 16:51:42: Joing agent: pred, thread: 0, round: 0
# 16:51:44: Finish population: 0, agent: pred, round: 0
# 16:51:44: Joing agent: pred, thread: 1, round: 0
# 16:51:44: Finish population: 1, agent: pred, round: 0
# 16:51:44: Finish Agent: pred, round: 0
# 16:51:44: Joing round: 0, thread: 1
# --------------------------------------------------
# 16:51:44: Start round #:1
# 16:51:44: Agent: pred, round: 1
# 16:51:44: Agent: prey, round: 1
# 16:51:44: Joing round: 1, thread: 0
# 16:51:45: population: 0, agent: prey, round: 1
# 16:51:45: population: 1, agent: prey, round: 1
# 16:51:45: Joing agent: prey, thread: 0, round: 1
# 16:51:47: Finish population: 0, agent: prey, round: 1
# 16:51:47: Joing agent: prey, thread: 1, round: 1
# 16:51:47: Finish population: 1, agent: prey, round: 1
# 16:51:47: Finish Agent: prey, round: 1
# 16:51:49: population: 0, agent: pred, round: 1
# 16:51:49: population: 1, agent: pred, round: 1
# 16:51:49: Joing agent: pred, thread: 0, round: 1
# 16:51:51: Finish population: 0, agent: pred, round: 1
# 16:51:51: Finish population: 1, agent: pred, round: 1
# 16:51:51: Joing agent: pred, thread: 1, round: 1
# 16:51:51: Finish Agent: pred, round: 1
# 16:51:51: Joing round: 1, thread: 1
# --------------------------------------------------
# 16:51:51: Start round #:2
# 16:51:51: Agent: pred, round: 2
# 16:51:51: Agent: prey, round: 2
# 16:51:51: Joing round: 2, thread: 0
# 16:51:52: population: 0, agent: prey, round: 2
# 16:51:52: population: 1, agent: prey, round: 2
# 16:51:52: Joing agent: prey, thread: 0, round: 2
# 16:51:54: Finish population: 0, agent: prey, round: 2
# 16:51:54: Joing agent: prey, thread: 1, round: 2
# 16:51:54: Finish population: 1, agent: prey, round: 2
# 16:51:54: Finish Agent: prey, round: 2
# 16:51:56: population: 0, agent: pred, round: 2
# 16:51:56: population: 1, agent: pred, round: 2
# 16:51:56: Joing agent: pred, thread: 0, round: 2
# 16:51:58: Finish population: 0, agent: pred, round: 2
# 16:51:58: Joing agent: pred, thread: 1, round: 2
# 16:51:58: Finish population: 1, agent: pred, round: 2
# 16:51:58: Finish Agent: pred, round: 2
# 16:51:58: Joing round: 2, thread: 1
# --------------------------------------------------

