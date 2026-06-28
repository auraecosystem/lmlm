const controller = new AbortController();

async function request(url: string) {
  const response = await fetch(url, {
    signal: controller.signal,
  });

  return response.json();
}

const requests = [
  request("/api/users"),
  request("/api/posts"),
  request("/api/comments"),
];

// Abort every request using the same controller.
controller.abort();

Promise.allSettled(requests).then(console.log);
