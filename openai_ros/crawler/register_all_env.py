from gym.envs.registration import register

register(
        id='CrawlerTestEnv-v0',
        entry_point='crawler.test_env:TestTaskEnv',
        kwargs={'robot_id': 0}
    )

register(
        id='CrawlerStandupEnv-v0',
        entry_point='crawler.standup_env:StandupTaskEnv',
        kwargs={'robot_id': 0}
    )

register(
        id='CrawlerWalkXEnv-v0',
        entry_point='crawler.walk_forward:WalkXTaskEnv',
        kwargs={'number': 1}
    )