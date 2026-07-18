"""冒烟测试：验证包可导入、版本号存在。"""


def test_package_importable():
    import agent_mem

    assert isinstance(agent_mem.__version__, str)
    assert agent_mem.__version__  # 非空


def test_subpackages_importable():
    import agent_mem.agent
    import agent_mem.kv
    import agent_mem.middleware
    import agent_mem.scheduler
    import agent_mem.server

    # 仅校验可导入，无副作用
    assert agent_mem.agent is not None
