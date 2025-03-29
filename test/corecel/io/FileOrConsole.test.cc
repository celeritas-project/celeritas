//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/FileOrConsole.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/FileOrConsole.hh"

#include <fstream>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
// FIXTURES
//---------------------------------------------------------------------------//

class FileOrConsoleTest : public ::celeritas::test::Test
{
  protected:
    void TearDown() override
    {
        // Clean up any unique files that might have been created
        for (auto const& fn : created_files_)
        {
            try
            {
                std::remove(fn.c_str());
            }
            catch (std::exception const& e)
            {
                FAIL() << e.what();
            }
        }
    }

    std::vector<std::string> created_files_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(FileOrConsoleTest, in)
{
    auto const filename = this->make_unique_filename(".txt");
    {
        // Create a test file with sample content
        std::ofstream test_file(filename);
        test_file << "test content" << std::endl;
        created_files_.push_back(filename);
    }

    // Test with a real file
    {
        FileOrStdin input(filename);
        EXPECT_EQ(input.filename(), filename);

        std::string content;
        std::istream& stream = input;
        std::getline(stream, content);
        EXPECT_EQ(content, "test content");
    }

    // Test with "-" (stdin)
    {
        FileOrStdin input("-");
        EXPECT_EQ(input.filename(), "<stdin>");
        // Note: We can't easily test reading from stdin in a unit test
        // without redirecting stdin, which is challenging in a portable way
    }
}

TEST_F(FileOrConsoleTest, out_overwrite)
{
    auto const filename = this->make_unique_filename(".txt");

    // Create a file first to test overwrite
    {
        std::ofstream pre_file(filename);
        pre_file << "old content" << std::endl;
        created_files_.push_back(filename);
    }

    // Test overwrite mode
    {
        FileOrStdout output(filename, FileOrStdout::OpenMode::overwrite);
        EXPECT_EQ(output.filename(), filename);

        std::ostream& stream = output;
        stream << "new content" << std::endl;
    }

    // Verify content was overwritten
    std::ifstream verify(filename);
    std::string content;
    std::getline(verify, content);
    EXPECT_EQ(content, "new content");
}

TEST_F(FileOrConsoleTest, out_error)
{
    auto const filename = this->make_unique_filename(".txt");

    // Create a file first
    {
        std::ofstream pre_file(filename);
        pre_file << "existing content" << std::endl;
        pre_file.close();
    }

    // Test error_if_exists mode - should throw
    EXPECT_THROW(
        {
            FileOrStdout output(filename,
                                FileOrStdout::OpenMode::error_if_exists);
        },
        RuntimeError);
}

TEST_F(FileOrConsoleTest, out_unique)
{
    auto const orig_filename = "test_output_unique.txt";
    try
    {
        // Just in case the file already existed from a previous test that
        // broke...
        std::remove(orig_filename);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(warning) << e.what();
    }

    // Test unique mode
    for (int i = 0; i < 4; ++i)
    {
        ScopedLogStorer scoped_log(&world_logger());
        FileOrStdout output(orig_filename, FileOrStdout::OpenMode::unique);

        // The filename should have been changed to something unique
        created_files_.push_back(output.filename());
        static_cast<std::ostream&>(output)
            << "unique: " << output.filename() << std::endl;

        if (i == 0)
        {
            EXPECT_EQ(orig_filename, output.filename());

            EXPECT_TRUE(scoped_log.empty());
        }
        else
        {
            EXPECT_NE(orig_filename, output.filename());
            EXPECT_TRUE(output.filename().find("test_output_unique")
                        != std::string::npos);

            static char const* const expected_log_levels[] = {"warning"};
            EXPECT_VEC_EQ(expected_log_levels, scoped_log.levels());

            // Note that the extension is unique so we can only test the front
            // of the warning
            ASSERT_EQ(1, scoped_log.messages().size());
            EXPECT_TRUE(starts_with(
                scoped_log.messages().front(),
                R"(Failed to open file 'test_output_unique.txt' without clobbering: renamed to test_output_unique-)"));
        }
    }

    // Verify all files exist
    for (auto const& filename : created_files_)
    {
        std::ifstream infile(filename);
        ASSERT_TRUE(infile.good());

        std::ostringstream expected;
        expected << "unique: " << filename;

        std::string actual;
        std::getline(infile, actual);
        EXPECT_EQ(expected.str(), actual);
    }
}

TEST_F(FileOrConsoleTest, out_stdout)
{
    // Test with "-" (stdout)
    FileOrStdout output("-", FileOrStdout::OpenMode::overwrite);
    EXPECT_EQ(output.filename(), "<stdout>");
    // Note: We can't easily test writing to stdout in a unit test

    // Test cast to pointer
    std::ostream* stream_ptr = output;
    EXPECT_EQ(stream_ptr, &std::cout);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
