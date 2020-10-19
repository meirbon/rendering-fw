#pragma once

#include <GL/glew.h>
#include "check.h"

namespace rfw::utils
{
static void draw_quad()
{
	static GLuint VAO = 0;
	if (!VAO)
	{
		// generate buffers
		static const GLfloat verts[] = {-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0};
		static const GLfloat uvdata[] = {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0};

		GLuint vertexBuffer;
		glGenBuffers(1, &vertexBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

		GLuint UVBuffer;
		glGenBuffers(1, &UVBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, UVBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvdata), uvdata, GL_STATIC_DRAW);

		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, (const void *)0);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, UVBuffer);
		glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, (const void *)0);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		CheckGL();
	}
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
	CheckGL();
}
} // namespace rfw::utils