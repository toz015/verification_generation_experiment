"""Bounded submission: a failed request stops new work from being scheduled."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def parallel_map(function, items, workers):
    if isinstance(workers,bool) or not isinstance(workers,int) or workers<1:
        raise ValueError('workers must be a positive integer')
    items=list(items)
    output=[None]*len(items)
    remaining=iter(enumerate(items))
    pool=ThreadPoolExecutor(max_workers=workers)
    pending={}
    try:
        for _ in range(min(workers,len(items))):
            index,item=next(remaining)
            pending[pool.submit(function,item)]=index
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            # Check every completed result before submitting another request.
            for future in done:
                output[pending.pop(future)]=future.result()
            for _ in done:
                next_item=next(remaining,None)
                if next_item is not None:
                    index,item=next_item
                    pending[pool.submit(function,item)]=index
        return output
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
