import pybullet
from pybullet_utils import bullet_client
import pkgutil


def connect(gui=1):
    if gui:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        # egl = pkgutil.get_loader("eglRenderer")
        # if egl:
        #     p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # else:
        #     p.loadPlugin("eglRendererPlugin")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    return p
