/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RestServer.h"

#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "crow.h"
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif

struct cudaq::RestServer::impl {
  crow::SimpleApp app;
};

cudaq::RestServer::RestServer(int port, const std::string &name) {
  m_impl = std::make_unique<impl>();
  m_impl->app.port(port);
  m_impl->app.server_name(name);
}
void cudaq::RestServer::start() { m_impl->app.run(); }
void cudaq::RestServer::stop() { m_impl->app.stop(); }
cudaq::RestServer::~RestServer() = default;
void cudaq::RestServer::addRoute(Method routeMethod, const char *route,
                                 RouteHandler handler) {
  switch (routeMethod) {
  case (Method::GET):
    m_impl->app.route_dynamic(route).methods("GET"_method)(
        [handler](const crow::request &req) {
          return handler(req.body).dump();
        });
    break;
  case (Method::POST):
    m_impl->app.route_dynamic(route).methods("POST"_method)(
        [handler](const crow::request &req) {
          return handler(req.body).dump();
        });
    break;
  }
}