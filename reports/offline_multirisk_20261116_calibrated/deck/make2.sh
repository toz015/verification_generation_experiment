set -e
cd "$(dirname "$0")"
export V=$1
SK="/Users/wanghd/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/2784a202-891f-4a36-8fc3-cc77f6abec77/bcafe2b9-c578-4e7c-9f87-e22cea087e98/skills/pptx"
PPTX_SKILL="$SK" DATA=$PWD/data_$V.json NODE_PATH=$PWD/node_modules node build.js $PWD/base.pptx
./venv/bin/python post.py base.pptx deck_$V.pptx
./venv/bin/python "$SK/scripts/office/validate.py" deck_$V.pptx
rm -rf render_$V && mkdir render_$V
osascript <<EOF
tell application "Keynote"
  set d to open (POSIX file "$PWD/deck_$V.pptx")
  delay 3
  export d to (POSIX file "$PWD/render_$V") as slide images with properties {image format:PNG}
  close d saving no
end tell
EOF
./venv/bin/python - <<'EOF'
from PIL import Image
import glob
fs=sorted(f for f in glob.glob('render_'+__import__('os').environ['V']+'/*.png') if 'sheet' not in f)
for g in range(0,len(fs),4):
    ims=[Image.open(f).convert('RGB') for f in fs[g:g+4]]
    w,h=ims[0].size
    sheet=Image.new('RGB',(2*w+30,2*h+30),'#888')
    for i,im in enumerate(ims):
        sheet.paste(im,((i%2)*(w+10)+10,(i//2)*(h+10)+10))
    sheet.save('render_'+__import__('os').environ['V']+f'/sheet{g//4+1}.png')
print(len(fs),'slides rendered')
EOF
