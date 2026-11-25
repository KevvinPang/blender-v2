"""Run the real bridge, then describe only the animation/camera import payload."""
import json,sys
from pathlib import Path
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'blender_bridge'))
from calibrated_bridge import main
main()
args=sys.argv[sys.argv.index('--')+1:]
output=Path(args[1]);rig=bpy.data.objects[args[args.index('--rig-name')+1]]
s=bpy.context.scene;camera=s.camera;chain=[]
while camera:
    chain.append(camera.name);camera=camera.parent
constraints={b.name:[{'name':c.name,'mute':c.mute} for c in b.constraints] for b in rig.pose.bones}
manifest={'blend':str(output),'action':rig.animation_data.action.name,'cameras':chain,
    'camera':s.camera.name,'frames':s.frame_end,'fps':s.render.fps,'fps_base':s.render.fps_base,
    'resolution':[s.render.resolution_x,s.render.resolution_y],
    'diagnostics':str(output.with_suffix('.diagnostics.npz')),'constraints':constraints,
    'hand_tracks':str(output.with_suffix('.hand_tracks.npz')),
    'rotation_modes':{b.name:b.rotation_mode for b in rig.pose.bones},
    'validation':s.get('mocap_v2_validation','{}')}
output.with_suffix('.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('V2_PACKAGE_READY',output,flush=True)
