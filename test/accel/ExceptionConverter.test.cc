//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ExceptionConverter.test.cc
//---------------------------------------------------------------------------//
#include "accel/ExceptionConverter.hh"

#include <G4VExceptionHandler.hh>

#include "corecel/Macros.hh"
#include "corecel/sys/Environment.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class ExceptionHandler final : public G4VExceptionHandler
{
  public:
    //! Accept error codes from geant4
    G4bool Notify(char const* originOfException,
                  char const* exceptionCode,
                  G4ExceptionSeverity severity,
                  char const* description) final
    {
        if (originOfException)
            origin = originOfException;
        if (exceptionCode)
            code = exceptionCode;
        if (description)
            desc = description;
        level = severity;

        return /* abort_the_program = */ false;
    }

    void clear()
    {
        origin = {};
        code = {};
        level = G4ExceptionSeverity::JustWarning;
        desc = {};
    }

    std::string origin;
    std::string code;
    G4ExceptionSeverity level;
    std::string desc;
};

class ExceptionConverterTest : public ::celeritas::test::Test
{
  protected:
    static ExceptionHandler& handler()
    {
        // Instantiating the error handler's base class changes the global
        // state
        static ExceptionHandler eh;
        return eh;
    }

    static void SetUpTestCase()
    {
        // Instantiate error handler
        handler();

        // Set environment variable to strip the source directory name from
        // assertions
        celeritas::environment().insert({"CELER_STRIP_SOURCEDIR", "1"});
    }

    void SetUp() override { handler().clear(); }
};

TEST_F(ExceptionConverterTest, debug)
{
    ExceptionConverter call_g4e{"test001"};
    CELER_TRY_HANDLE(
        throw ::celeritas::DebugError({::celeritas::DebugErrorType::internal,
                                       "there are five lights",
                                       "me.cc",
                                       123}),
        call_g4e);

    EXPECT_EQ("me.cc:123", handler().origin);
    EXPECT_EQ("test001", handler().code);
    EXPECT_EQ(G4ExceptionSeverity::FatalException, handler().level);
    EXPECT_EQ("internal assertion failed: there are five lights",
              handler().desc);
}

TEST_F(ExceptionConverterTest, runtime)
{
    ExceptionConverter call_g4e{"test002"};
    CELER_TRY_HANDLE(CELER_VALIDATE(2 + 2 == 5, << "math actually works"),
                     call_g4e);

    EXPECT_EQ(0, handler().origin.find("accel/ExceptionConverter.test.cc"))
        << "actual path: '" << handler().origin << '\'';
    EXPECT_EQ("test002", handler().code);
    EXPECT_EQ("math actually works", handler().desc);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
