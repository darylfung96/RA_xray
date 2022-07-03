def increase_by_percentage(content, percentage):
	lines = content.split("\n")[:-1]
	lines = [line.split() for line in lines]

	for i in range(len(lines)):
		line = lines[i]
		line[3] = str(float(line[3]) * percentage)
		line[4] = str(float(line[4]) * percentage)

	lines = [' '.join(line) for line in lines]
	lines = '\n'.join(lines)

	return lines

content = """"""
print(increase_by_percentage(content, 1.20))