import sys
from heapq import heappush as push, heappop as pop

def print_hello (string):
    print ("hi "+ string)

# use heap to get median on many numbers
def medians_heap(numbers_in):
    numbers = iter(numbers_in)
    less, more = [], []
    first = next(numbers)
    yield first
    second = next(numbers)
    push(less, - min(first, second))
    push(more, max(first, second))
    while True:
        current = ( more[0] if len(less) < len(more)
                    else - less[0] if len(less) > len(more)
                    else (more[0] - less[0]) / 2 )
        yield current
        number = next(numbers)
        if number <= current:
            push(less, - number)
        else:
            push(more, number)
        small, big = ((less, more) if len(less) <= len(more)
                      else (more, less))
        if len(big) - len(small) > 1: push(small, - pop(big))

if __name__ == '__main__':
    #infile = sys.argv[1]
        #    depths = []
    *_, last = medians_heap( (3,74, 1,2,4))
    print (last)
    #for median in foo:
    #    pass

    print("hello")
    depths = [3, 74, 4, 1, 2]
    medians_heap ([])
    print(depths)
    print_hello ("Pauline")
    median = medians_heap(depths)
    print("done")
    print (median)
    print(next(median))
    print ("printed median")