
What would be the output in this python code?
```python
def my_function(mydict:dict):
    mydict['key_0'] +=1 # you could do if isinstance(mydict, dict)
    mydict = 2          # you could use mydict.get('key_0')

    return mydict

if __name__ == '__main__':
    mydict = {'key_0': 0}

    out = my_function(mydict)
    print(out)
```
