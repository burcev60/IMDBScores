# IMDBScores

## Обзор

Веб-приложение IMDBScores создано на Django для предсказания оценок и стилистической окраски (позитивной или негативной) отзывов с сайта IMDB. Приложение представляет собой одностраничный интерфейс, который позволяет пользователям вводить комментарии и получать мгновенную обратную связь.

## Особенности

- **Одностраничное приложение**: Чистый и интуитивно понятный интерфейс, где пользователи могут вводить свои комментарии и получать предсказания.
- **Интеграция AJAX**: Отправка данных на сервер без обновления страницы, что обеспечивает плавный пользовательский опыт.
- **Предобработка данных**: Отзывы, отправленные пользователями, проходят такую же предобработку, как и обучающая выборка, чтобы обеспечить согласованность в предсказаниях.
- **Векторизация TF-IDF**: Использует алгоритм TF-IDF для преобразования текстовых данных в формат, подходящий для ввода модели.
- **Предсказание модели**: Предобученная модель генерирует оценку для отзыва и определяет его стилистическую окраску на основе предсказанной оценки.
