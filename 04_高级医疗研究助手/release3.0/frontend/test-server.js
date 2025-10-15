const http = require('http');

const options = {
  hostname: 'localhost',
  port: 3000,
  path: '/',
  method: 'GET',
  timeout: 5000
};

const req = http.request(options, (res) => {
  console.log(`状态码: ${res.statusCode}`);
  console.log('响应头:', res.headers);

  if (res.statusCode === 200) {
    console.log('✅ 前端服务运行正常');
  } else {
    console.log('❌ 前端服务返回异常状态码:', res.statusCode);
  }
});

req.on('error', (e) => {
  console.log('❌ 无法连接到前端服务:', e.message);
});

req.on('timeout', () => {
  console.log('❌ 连接前端服务超时');
  req.destroy();
});

req.end();
