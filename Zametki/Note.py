class Note:
    def __init__(self, note_id, title, body, timestamp):
        self.note_id = note_id
        self.title = title
        self.body = body
        self.timestamp = timestamp

    # Строка для распечатки записей
    def __str__(self):
        line = f"ID: {self.note_id},\tЗаголовок: {self.title},\tТело: {self.body},\tВремя: {self.timestamp}"
        return line
