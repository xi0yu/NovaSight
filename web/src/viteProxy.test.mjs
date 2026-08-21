// @vitest-environment node

import { createServer as createHttpServer, request as httpRequest } from "node:http";

import { afterEach, describe, expect, test } from "vitest";
import { createServer as createViteServer } from "vite";

import viteConfig from "../vite.config";

const servers = [];

afterEach(async () => {
  await Promise.allSettled(servers.splice(0).map((server) => closeServer(server)));
});

describe("Vite API proxy", () => {
  test("preserves the browser Host across every proxied surface", () => {
    for (const route of ["/api", "/healthz", "/ws"]) {
      expect(viteConfig.server.proxy[route]).toMatchObject({ changeOrigin: false });
    }
  });

  test("preserves the browser Host for backend Origin validation", async () => {
    let observedHost;
    let observedOrigin;
    const backend = createHttpServer((request, response) => {
      observedHost = request.headers.host;
      observedOrigin = request.headers.origin;

      const requestHost = new URL(`http://${observedHost}`).hostname;
      const originHost = new URL(observedOrigin).hostname;
      const accepted = requestHost === originHost;
      response.writeHead(accepted ? 200 : 403, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ code: accepted ? "OK" : "ORIGIN_REJECTED" }));
    });
    servers.push(backend);
    const backendAddress = await listen(backend);

    const configuredProxy = viteConfig.server.proxy["/api"];
    expect(configuredProxy).toBeDefined();
    const proxy =
      typeof configuredProxy === "string"
        ? `http://127.0.0.1:${backendAddress.port}`
        : {
            ...configuredProxy,
            target: `http://127.0.0.1:${backendAddress.port}`,
          };
    const vite = await createViteServer({
      configFile: false,
      appType: "custom",
      server: {
        middlewareMode: true,
        proxy: { "/api": proxy },
      },
    });
    const frontend = createHttpServer(vite.middlewares);
    servers.push(frontend);
    const frontendAddress = await listen(frontend);

    let response;
    try {
      response = await sendLogin(frontendAddress.port);
    } finally {
      await vite.close();
    }

    expect(response).toEqual({ status: 200, body: '{"code":"OK"}' });
    expect(observedHost).toBe("100.64.0.42:7351");
    expect(observedOrigin).toBe("http://100.64.0.42:7351");
  });
});

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve(server.address());
    });
  });
}

function closeServer(server) {
  return new Promise((resolve) => {
    if (!server.listening) {
      resolve();
      return;
    }
    server.close(() => resolve());
  });
}

function sendLogin(port) {
  return new Promise((resolve, reject) => {
    const request = httpRequest(
      {
        hostname: "127.0.0.1",
        port,
        path: "/api/auth/session",
        method: "POST",
        headers: {
          Host: "100.64.0.42:7351",
          Origin: "http://100.64.0.42:7351",
          "Content-Type": "application/json",
        },
      },
      (response) => {
        response.setEncoding("utf8");
        let body = "";
        response.on("data", (chunk) => {
          body += chunk;
        });
        response.on("end", () => resolve({ status: response.statusCode, body }));
      },
    );
    request.once("error", reject);
    request.end('{"access_code":"test"}');
  });
}
