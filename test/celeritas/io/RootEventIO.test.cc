//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootEventIO.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/io/RootEventReader.hh"
#include "celeritas/io/RootEventWriter.hh"

#include "EventIOTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class EventIOTest : public EventIOTestBase
{
};

TEST_F(EventIOTest, write_read)
{
    std::string filename = this->make_unique_filename(std::string{".root"});

    // Write events
    {
        RootEventWriter write_event(filename, this->particles());
        this->write_test_event(std::ref(write_event));
    }

    RootEventReader read_event(filename, this->particles());
    this->read_check_test_event(std::ref(read_event));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
