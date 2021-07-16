# monsters-corporation

- Использование детектора эмоций по лицу человека для распозвавания смеха
- Создание интерактивной игры по идее мультфильма "Корпорация монстров" - уровни с накоплением энергии. 
- (усложнение) Добавить распознавание смеха по аудио



Architecture:
    HEAD-script [Python]
        - real-time video 
        - smile, laugh detector
        - metrics: 
            - percentage bar
            - score
        - fixed time    
    - telegram requests showing funny images
    - score recorded to the db


