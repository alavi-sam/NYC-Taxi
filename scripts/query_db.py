import duckdb
import sys
import os


if __name__=='__main__':
    con = duckdb.connect('taxi_data.db')
    print(">>", end='')
    query = sys.stdin.read()

    while query:
        try:
            res = con.execute(query).df()   
            print(res)
            print(">>", end='')
            query = sys.stdin.read()
        except Exception as e:
            os.system('select 1;')
            print(str(e))
            print(">>", end='')
            query = sys.stdin.read()

        if query == '/q':
            break

    con.close()
    
