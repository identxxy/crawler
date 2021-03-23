from gym.envs.registration import register

register(
        id='CrawlerTestEnv-v0',
        entry_point='crawler.test_env:TestTaskEnv',
        kwargs={'n': 1, 
                'displacement_xyz': [0, 10, 0]}
    )

register(
        id='CrawlerStandupEnv-v0',
        entry_point='crawler.standup_env:StandupTaskEnv',
        kwargs={'n': 1, 
                'displacement_xyz': [0, 10, 0]}
    )

register(
        id='CrawlerWalkXEnv-v0',
        entry_point='crawler.walk_forward:WalkXTaskEnv',
        kwargs={'n': 1, 
                'displacement_xyz': [0, 10, 0]}
    )

register(
        id='CrawlerWalkXEnv-v1',
        entry_point='crawler.walk_forward:WalkXTaskEnv_v1',
        kwargs={'n': 1, 
                'displacement_xyz': [0, 10, 0]}
    )

register(
        id='CrawlerWalkXCamEnv-v0',
        entry_point='crawler.walk_forward_cam:WalkXTaskEnv_v0',
        kwargs={'n': 1, 
                'displacement_xyz': [0, 10, 0]}
)

register(
        id='CrawlerWalkXCamEnv-v1',
        entry_point='crawler.walk_forward_cam:WalkXTaskEnv_v1',
        kwargs={'n': 1, 
                'displacement_xyz': [0, 10, 0]}
)