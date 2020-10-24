from gym.envs.registration import register

register(
        id='CrawlerTestEnv-v0',
        entry_point='crawler.test_env:TestTaskEnv'
    )
