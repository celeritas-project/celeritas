//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Environment.cc
//---------------------------------------------------------------------------//
#include "Environment.hh"

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Access a static global environment variable.
 *
 * This static variable should be shared among Celeritas objects.
 */
Environment& environment()
{
    static Environment environ;
    return environ;
}

//---------------------------------------------------------------------------//
/*!
 * Thread-safe access to global modified environment variables.
 */
const std::string& getenv(const std::string& key)
{
    static std::mutex           s_getenv_mutex;
    std::lock_guard<std::mutex> scoped_lock{s_getenv_mutex};
    return environment()[key];
}

//---------------------------------------------------------------------------//
/*!
 * Write the accessed environment variables to a stream.
 */
std::ostream& operator<<(std::ostream& os, const Environment& env)
{
    os << "{\n";
    for (const auto& kv : env)
    {
        os << "  " << kv.first << ": '" << kv.second << "',\n";
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
// MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Set a value from the system environment.
 */
auto Environment::load_from_getenv(const key_type& key) -> const mapped_type&
{
    std::string value;
    if (const char* sys_value = std::getenv(key.c_str()))
    {
        // Variable is set in the user environment
        value = sys_value;
    }
    auto iter_inserted = vars_.emplace(key, std::move(value));
    CELER_ASSERT(iter_inserted.second);
    return iter_inserted.first->second;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
