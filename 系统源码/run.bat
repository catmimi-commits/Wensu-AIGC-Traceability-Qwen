@echo off
chcp 65001 >nul
echo 正在启动 闻溯——AIGC新闻文本双轨制溯源与治理系统...
echo.

:: 安装依赖
echo [1/2] 检查并安装依赖...
pip install -r backend/requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 启动Flask
echo [2/2] 启动后端服务...
echo.
echo 启动成功！请在浏览器访问：http://localhost:5000
echo 按 Ctrl+C 可停止服务
echo.
cd backend
python app.py
pause
