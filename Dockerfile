# Sử dụng ảnh base là python:3.9
FROM python:3.9

# Tạo thư mục làm việc /app và thiết lập làm thư mục làm việc mặc định
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Command để chạy ứng dụng
CMD ["python", "main.py"]
