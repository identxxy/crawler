from gym.envs.registration import register

register(
        id='CrawlerTestEnv-v0',
        entry_point='crawler.test_env:TestTaskEnv'
    )

register(
        id='CrawlerStandupEnv-v0',
        entry_point='crawler.standup_env:StandupTaskEnv'
    )