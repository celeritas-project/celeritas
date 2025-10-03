//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/JsonEventIO.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/io/JsonEventReader.hh"
#include "celeritas/io/JsonEventWriter.hh"

#include "EventIOTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class JsonEventIOTest : public EventIOTestBase
{
};

TEST_F(JsonEventIOTest, write_read)
{
    std::string filename = this->make_unique_filename(".jsonl");

    // Write events
    {
        JsonEventWriter write_event(filename, this->particles());
        this->write_test_event(std::ref(write_event));
    }

    JsonEventReader reader(filename, this->particles());
    EXPECT_EQ(3, reader.num_events());
    this->read_check_test_event(reader);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
