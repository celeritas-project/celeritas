//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file detail/CliCommon.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CLI/CLI.hpp>

namespace celeritas
{
namespace app
{
namespace detail
{
//---------------------------------------------------------------------------//

inline std::string failure_message(const CLI::App* cli, const CLI::Error& e)
{
    std::ostringstream os;
    os << cli->get_name() << ": " << CLI::FailureMessage::simple(cli, e);
    return std::move(os).str();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace app
}  // namespace celeritas
