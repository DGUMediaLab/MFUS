#pragma once

#include <Stdio.h>

#include <String>
using namespace std;

typedef unsigned char BYTE;

class BinaryStream
{
public:
	FILE* m_pFile;

public:
	BinaryStream() : m_pFile(0) { }
	virtual ~BinaryStream()
	{
		CloseFile();
	}

	bool OpenFile(const string& FileName, const string& mode)
	{
		CloseFile();

		fopen_s(&m_pFile, FileName.c_str(), mode.c_str());

		if (m_pFile == 0)
		{
			return false;
		}

		return true;
	}

	void CloseFile(void)
	{
		if (m_pFile != 0)
		{
			fclose(m_pFile);

			m_pFile = 0;
		}
	}
};

class BinaryReader : public BinaryStream
{
public:
	BinaryReader(const string& FileName)
	{
		OpenFile(FileName, "rb");
	}
	virtual ~BinaryReader()
	{
		CloseFile();
	}

	char ReadChar(void)
	{
		if (m_pFile == 0)
		{
			// Error
		}

		char data;
		fread(&data, sizeof(char), 1, m_pFile);

		return data;
	}

	BYTE ReadBYTE(void)
	{
		if (m_pFile == 0)
		{
			// Error
		}

		BYTE data;
		fread(&data, sizeof(BYTE), 1, m_pFile);

		return data;
	}

	BYTE* ReadBYTEs(unsigned int size)
	{
		if (m_pFile == 0)
		{
			// Error
		}

		BYTE* data = new BYTE[size];
		fread(data, sizeof(BYTE), size, m_pFile);

		return data;
	}

	short ReadInt16(void)
	{
		if (m_pFile == 0)
		{
			// Error
		}

		short data;
		fread(&data, sizeof(short), 1, m_pFile);

		return data;
	}

	int ReadInt32(void)
	{
		if (m_pFile == 0)
		{
			// Error
		}

		int data;
		fread(&data, sizeof(int), 1, m_pFile);

		return data;
	}

	unsigned int ReadUInt32(void)
	{
		if (m_pFile == 0)
		{
			// Error
		}

		unsigned int data;
		fread(&data, sizeof(unsigned int), 1, m_pFile);

		return data;
	}

	string ReadString(void)
	{
		if (m_pFile == 0)
		{
			// Error
		}

		string data = "";
		char ch;
		while (!feof(m_pFile))
		{
			fread(&ch, sizeof(char), 1, m_pFile);

			if (ch == '\0')
			{
				break;
			}

			data += ch;
		};

		return data;
	}
};

class BinaryWriter : public BinaryStream
{
public:
	BinaryWriter(const string& FileName)
	{
		OpenFile(FileName, "wb");
	}
	virtual ~BinaryWriter()
	{
		CloseFile();
	}

	bool Write(char data)
	{
		if (m_pFile == 0)
		{
			return false;
		}

		fwrite(&data, sizeof(char), 1, m_pFile);

		return true;
	}

	bool Write(BYTE data)
	{
		if (m_pFile == 0)
		{
			return false;
		}

		fwrite(&data, sizeof(BYTE), 1, m_pFile);

		return true;
	}

	bool Write(BYTE* data, unsigned int size)
	{
		if (m_pFile == 0)
		{
			return false;
		}

		fwrite(&data, sizeof(BYTE), size, m_pFile);

		return true;
	}

	bool Write(int data)
	{
		if (m_pFile == 0)
		{
			return false;
		}

		fwrite(&data, sizeof(int), 1, m_pFile);

		return true;
	}

	bool Write(const string& data)
	{
		if (m_pFile == 0)
		{
			return false;
		}

		fwrite(data.c_str(), sizeof(char), data.length() + 1, m_pFile);

		return true;
	}
};