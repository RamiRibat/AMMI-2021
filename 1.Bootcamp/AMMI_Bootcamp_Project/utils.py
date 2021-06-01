


def reader(filepath, headers):
  f = open(filepath, "r")
  # Columns = ['ID', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'class']
  list_dic = []
  for line in f:
    line = [float(x) for x in line.split(',')]
    dic = {}
    for i in range (len(headers)) :
      dic[headers[i]] = line[i]
    list_dic.append(dic)
  return(list_dic)
