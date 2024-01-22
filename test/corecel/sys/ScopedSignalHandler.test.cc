//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedSignalHandler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/ScopedSignalHandler.hh"

#include <csignal>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(ScopedSignalHandlerTest, single)
{
    {
        ScopedSignalHandler interrupted(SIGINT);
        EXPECT_TRUE(interrupted);
        EXPECT_FALSE(interrupted());
        std::raise(SIGINT);
        EXPECT_TRUE(interrupted());
    }
    {
        ScopedSignalHandler interrupted(SIGINT);
        EXPECT_TRUE(interrupted);
        EXPECT_FALSE(interrupted());
        std::raise(SIGINT);
        EXPECT_TRUE(interrupted());
        interrupted = {};
        EXPECT_FALSE(interrupted());
        EXPECT_FALSE(interrupted);
    }
}

#ifndef _WIN32
TEST(ScopedSignalHandlerTest, multiple)
{
    using FuncPtr = int (*)(int);

    for (auto* raise :
         {FuncPtr(&ScopedSignalHandler::raise), FuncPtr(&std::raise)})
    {
        {
            ScopedSignalHandler interrupted{SIGINT, SIGUSR1};
            EXPECT_TRUE(interrupted);
            EXPECT_FALSE(interrupted());
            raise(SIGINT);
            EXPECT_TRUE(interrupted());
            raise(SIGUSR1);
            EXPECT_TRUE(interrupted());
        }
        {
            ScopedSignalHandler interrupted{SIGINT, SIGUSR1};
            raise(SIGUSR1);
            EXPECT_TRUE(interrupted());
        }
    }
}
#endif

TEST(ScopedSignalHandlerTest, nested)
{
    {
        ScopedSignalHandler interrupted(SIGINT);
        EXPECT_FALSE(interrupted());
        {
            // Nested of different type
            ScopedSignalHandler terminating(SIGTERM);
            EXPECT_FALSE(terminating());
            std::raise(SIGINT);
            EXPECT_TRUE(interrupted());
            EXPECT_FALSE(terminating());
            std::raise(SIGTERM);
            EXPECT_TRUE(interrupted());
            EXPECT_TRUE(terminating());
        }

        // Cannot create a new handler while an interrupt is unhandled for
        // that type
        EXPECT_THROW(ScopedSignalHandler(SIGINT), RuntimeError);

        // Clear outer interrupt
        interrupted = {};
        {
            ScopedSignalHandler also_interrupted(SIGINT);
            EXPECT_FALSE(also_interrupted());
            std::raise(SIGINT);
            EXPECT_TRUE(also_interrupted());
        }
    }
    {
        ScopedSignalHandler outer(SIGINT);
        EXPECT_FALSE(outer());
        {
            ScopedSignalHandler inner(SIGINT);
            EXPECT_FALSE(inner());
            std::raise(SIGINT);
            EXPECT_TRUE(inner());
            EXPECT_TRUE(outer());
        }
        // Inner destructor clears signal
        EXPECT_FALSE(outer());

        // Outer should still be able to handle signals though
        std::raise(SIGINT);
        EXPECT_TRUE(outer());
    }
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
